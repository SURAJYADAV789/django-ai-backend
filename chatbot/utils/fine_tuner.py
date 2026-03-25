import os 
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def upload_training_file(filepath: str) -> str:
    '''Upload you training data to OpenAI'''
    with open(filepath, 'rb') as f:
        response = client.files.create(file=f, purpose='fine-tune')
    print(f'File Uploaded: {response.id}')
    return response.id



def start_finetuning(file_id: str) -> str:
    '''Start the fine-tuning job'''
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini-2024-07-18",   # cheaper, fast and good quality
    )
    print(f'Fine-tuning started: {job.id}')
    return job.id

def check_status(jod_id: str) -> dict:
    '''Check is training is done'''
    job = client.fine_tuning.jobs.retrieve(jod_id)
    return {
        'status': job.status,
        'model': job.fine_tuned_model  # your new model name when done
    }
