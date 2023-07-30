import logging

import click
import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY


def load_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for GGML quantized models")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            max_ctx_size = 2048
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)

        else:
            # The code supports all huggingface models that ends with GPTQ and have some variation
            # of .no-act.order or .safetensors in their HF repo.
            logging.info("Using AutoGPTQForCausalLM for quantized models")

            if ".safetensors" in model_basename:
                # Remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            logging.info("Tokenizer loaded")

            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
    elif (
        device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
         task="text2text-generation", # maybe we try text-generation later on
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,  # keep this 0
        top_p=1, # keep this between 0.8 and 1
        repetition_penalty=1.15,
        generation_config=generation_config,
        top_k=80   # keep this above 50
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
def main(device_type, show_sources):
    """
    This function implements the information retrieval task.


    1. Loads an embedding model, can be HuggingFaceInstructEmbeddings or HuggingFaceEmbeddings
    2. Loads the existing vectorestore that was created by inget.py
    3. Loads the local LLM using load_model function - You can now set different LLMs.
    4. Setup the Question Answer retreival chain.
    5. Question answers.
    """

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")

    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})

    # uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()

    # load the LLM for generating Natural Language responses

    # for HF models
    # model_id = "arogov/llama2_13b_chat_uncensored"
    # model_id = "nkpz/llama2-22b-chat-wizard-uncensored"
    # model_basename = None

    # model_id = "TheBloke/Wizard-Vicuna-13B-Uncensored-HF"
    # model_id = "TheBloke/guanaco-7B-HF"
    # model_id = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM. Using STransformers
    # alongside will 100% create OOM on 24GB cards.
    # llm = load_model(device_type, model_id=model_id)

    # for GPTQ (quantized) models
    # model_id = "TheBloke/Nous-Hermes-13B-GPTQ"
    # model_basename = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
    # model_id = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
    # model_basename = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors" # Requires
    # ~21GB VRAM. Using STransformers alongside can potentially create OOM on 24GB cards.

    # model_id = "TheBloke/llama-2-13B-Guanaco-QLoRA-GGML"
    # model_basename = "llama-2-13b-guanaco-qlora.ggmlv3.q8_0.bin"

    model_id = "TheBloke/OpenAssistant-Llama2-13B-Orca-8K-3319-GGML"
    model_basename = "openassistant-llama2-13b-orca-8k-3319.ggmlv3.q8_0.bin"

    # model_id = "TheBloke/Luna-AI-Llama2-Uncensored-GGML"
    # model_basename = "luna-ai-llama2-uncensored.ggmlv3.q4_0.bin"

    # for GGML (quantized cpu+gpu+mps) models - check if they support llama.cpp
    # model_id = "TheBloke/wizard-vicuna-13B-GGML"
    # model_basename = "wizard-vicuna-13B.ggmlv3.q4_0.bin"
    # model_basename = "wizard-vicuna-13B.ggmlv3.q6_K.bin"
    # model_basename = "wizard-vicuna-13B.ggmlv3.q2_K.bin"

    # model_id = "TheBloke/Luna-AI-Llama2-Uncensored-GGML"
    # model_basename = "luna-ai-llama2-uncensored.ggmlv3.q8_0.bin"

    # model_id="TheBloke/Llama-2-13B-chat-GGML"
    # model_basename = "llama-2-13b-chat.ggmlv3.q6_K.bin"

    # model_id = "TheBloke/Llama-2-7B-Chat-GGML"
    #
    # model_basename = "llama-2-7b-chat.ggmlv3.q8_0.bin"

    # model_id = "TheBloke/llama2_7b_chat_uncensored-GGML"
    # model_basename = "llama2_7b_chat_uncensored.ggmlv3.q8_0.bin"

    model_id = "TheBloke/Llama-2-7B-Chat-GGML"
    model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"

    llm = load_model(device_type, model_id=model_id, model_basename=model_basename)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    question_list = ['In which article it is mentioned that a retired military personnel from the MOI can be '
                     'appointed in civil capcity?',
                     'Where was the Federal Decree-Law number 32 of 2019 concerning the civil service published? and '
                     'on which date?',
                     'Can a staff member be promoted in the MOI if he gas a disciplinary sanction against him?',
                     'I am an assistant undersecretary employee in the MOI, and I am going on an official mission to '
                     'Russia for 6 days.'
                     'What will be my total allowance for this mission?',
                     'If I go on an official mission with my family for 70 days, will the Ministry of the Interior (MOI)'
                     'in the UAE cover the expenses for treatment and medications?',
                     'List for me 6 reasons why an employee service can be terminated in the MOI.',
                     'Which place was the Federal Decree-Law number 32 of 2019 concerning the civil service issued at?',
                     'Summarize for me Article 71 of the Federal Decree-Law number 32 of 2019 of the MOI in the UAE.',
                     'Considering my functional grade is sixth grade, which ticket class will I be eligible for if I am traveling for an official mission outside the country?',
                     'Can non-staff members of the Ministry of Interior be sent on official missions outside the country?']

    print(len(question_list))

    # Interactive questions and answers
    for question in question_list:
        query = question  # input("\nEnter a query: ")
        # if query == "exit":
        #     break
        # Get the answer from the chain
        res = qa(query)
        print(res)  # next time we run
        answer, docs = res["result"], res["source_documents"]  # that is why we need to print this fully

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if show_sources:  # this is a flag that you can set to disable showing answers.
            # # Print the relevant sources used for the answer
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
