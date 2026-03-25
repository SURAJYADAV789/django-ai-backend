from django.core.management.base import BaseCommand
from chatbot.utils.fine_tuner import upload_training_file,start_finetuning,check_status

class Command(BaseCommand):
    help = 'Fine-tune openai model with your data'

    def add_arguments(self, parser):
        parser.add_argument('action', choices=['upload', 'train', 'status'])
        parser.add_argument('--file', type=str, help='File path or file id')
        parser.add_argument('--job-id', type=str, help='Job ID to check status')


    def handle(self, *args, **options):
        action = options['action']

        if action == 'upload':
            file_id = upload_training_file(options['file'])
            self.stdout.write(f'File ID: {file_id}')

        elif action == 'train':
            job_id = start_finetuning(options['file'])
            self.stdout.write(f'Job ID: {job_id}')

        elif action == 'status':
            result = check_status(options['job_id'])
            self.stdout.write(str(result))