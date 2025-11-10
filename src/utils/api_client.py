import requests
import json
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DocQAAPIClient:
    def __init__(self, api_endpoint: str, api_key: str, state_machine_arn: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.state_machine_arn = state_machine_arn
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key
        }
    
    def invoke_agent(self, query: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Invoke the DocQA agent orchestrator via Step Functions API
        """
        payload = {
            "stateMachineArn": self.state_machine_arn,
            "input": json.dumps({"query": query})
        }
        
        try:
            logger.info(f"Invoking DocQA API with query: {query[:100]}...")
            
            response = requests.post(
                self.api_endpoint,
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Parse the output from Step Functions
                if 'output' in result:
                    output_data = json.loads(result['output'])
                    return self._parse_agent_response(output_data, result)
                else:
                    logger.error(f"No output in API response: {result}")
                    return self._create_error_response("No output received from API")
                    
            else:
                logger.error(f"API call failed with status {response.status_code}: {response.text}")
                return self._create_error_response(f"API call failed: {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            return self._create_error_response("Request timed out. Please try again.")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return self._create_error_response(f"Network error: {str(e)}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            return self._create_error_response("Invalid response format")
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return self._create_error_response(f"Unexpected error: {str(e)}")
    
    def _parse_agent_response(self, output_data: Dict[str, Any], raw_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the agent response from Step Functions output
        """
        try:
            # Extract the main response fields
            status = output_data.get('status', 'UNKNOWN')
            final_answer = output_data.get('final_answer', 'No answer provided')
            citations = output_data.get('citations', [])
            routing_category = output_data.get('routing_category')
            categories = output_data.get('categories', [])
            
            # Extract execution metadata
            execution_time = 0
            if 'startDate' in raw_response and 'stopDate' in raw_response:
                execution_time = (raw_response['stopDate'] - raw_response['startDate']) * 1000
            
            billing_duration = raw_response.get('billingDetails', {}).get('billedDurationInMilliseconds', 0)
            
            return {
                'success': True,
                'status': status,
                'answer': final_answer,
                'citations': citations,
                'routing_category': routing_category,
                'categories': categories,
                'execution_time_ms': execution_time,
                'billing_duration_ms': billing_duration,
                'execution_arn': raw_response.get('executionArn'),
                'raw_response': output_data
            }
            
        except Exception as e:
            logger.error(f"Failed to parse agent response: {e}")
            return self._create_error_response("Failed to parse response")
    
    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """
        Create a standardized error response
        """
        return {
            'success': False,
            'status': 'ERROR',
            'answer': f"I apologize, but I encountered an error: {message}",
            'citations': [],
            'routing_category': None,
            'categories': [],
            'error': message
        }
    
    def health_check(self) -> bool:
        """
        Perform a simple health check with a greeting query
        """
        try:
            response = self.invoke_agent("hello", timeout=10)
            return response.get('success', False)
        except Exception:
            return False

class APIClientManager:
    """
    Singleton manager for API client instances
    """
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(APIClientManager, cls).__new__(cls)
        return cls._instance
    
    def get_client(self, api_endpoint: str, api_key: str, state_machine_arn: str) -> DocQAAPIClient:
        """
        Get or create an API client instance
        """
        if self._client is None or self._needs_refresh(api_endpoint, api_key, state_machine_arn):
            self._client = DocQAAPIClient(api_endpoint, api_key, state_machine_arn)
            self._last_config = (api_endpoint, api_key, state_machine_arn)
        
        return self._client
    
    def _needs_refresh(self, api_endpoint: str, api_key: str, state_machine_arn: str) -> bool:
        """
        Check if client needs to be refreshed due to config changes
        """
        if not hasattr(self, '_last_config'):
            return True
        
        return self._last_config != (api_endpoint, api_key, state_machine_arn)
