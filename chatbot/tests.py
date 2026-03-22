import random
from django.test import TestCase
from django.core.cache import cache
from unittest.mock import patch, MagicMock
# Create your tests here.

class RateLimitTest(TestCase):

    def setUp(self):
        cache.clear()

    def tearDown(self):
        cache.clear()


    def _post(self, ip='127.0.0.1', question='what is AI?'):
        '''Helper make a POST to /chat/ from given IP'''
        return self.client.post(
            '/chat/',
            data=b'{"question": "' + question.encode() + b'" }',
            content_type='application/json',
            REMOTE_ADDR=ip
        )
    
    #  Normal request succeeds
    @patch('chatbot.views.client')
    def test_valid_request_return_200(self, mock_client):
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Hello'))]
        )
        res = self._post(ip='10.0.0.1')
        self.assertEqual(res.status_code, 200)
        self.assertIn('answer', res.json())

    # Exceeding 10 request from same ip address
    @patch('chatbot.views.client')
    def test_rate_for_blocks_11th_request(self, mock_client):
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Answere'))]
        )
        ip = '10.0.0.2'
        for i in range(10):
            res = self._post(ip=ip)
            self.assertEqual(res.status_code, 200, f"Request {i+1} should succeed")

        blocked = self._post(ip=ip)
        self.assertEqual(blocked.status_code, 403)  # django ratelimit returns


    # Different IPs are rate-limited indenpendently
    @patch('chatbot.views.client')
    def test_different_ips_independent_limits(self, mock_client):
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Answere'))]
        )           
        # Exhaust IP A
        for _ in range(10):
            self._post(ip='1.1.1.1')

        # IP B still should works
        res = self._post(ip='2.2.2.2')
        self.assertEqual(res.status_code, 200)
    

    # Empty questions 400 not a rate limit issue
    @patch('chatbot.views.client')
    def test_empty_question_returns_400(self, mock_client):
        res = self.client.post(
            '/chat/',
            data=b'{"question": ""}',
            content_type='application/json',
            REMOTE_ADDR='5.5.5.5'
        )
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()['error'], 'No question provided')


    # Invalid Json
    def test_invalid_json_returns_400(self):
        res = self.client.post(
            '/chat/',
            data=b'not-json',
            content_type='application/json',
            REMOTE_ADDR= '6.6.6.6'
        )
        self.assertEqual(res.status_code, 400)


    # GET request -> 405
    def test_get_request_not_allowed(self):
        res = self.client.get('/chat/')
        self.assertEqual(res.status_code, 405)