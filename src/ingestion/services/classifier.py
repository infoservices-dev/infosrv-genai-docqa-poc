import logging
import json
from typing import Dict, List, Optional
import boto3
from botocore.exceptions import ClientError
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ingestion.config.settings import (
    CLASSIFICATION_TABLE_NAME,
    CLASSIFICATION_MODEL_ID,
    CLASSIFICATION_MAX_TOKENS,
    CLASSIFICATION_TEMPERATURE,
    AWS_REGION
)

logger = logging.getLogger(__name__)


class DocumentClassifier:
    
    def __init__(
        self,
        table_name: Optional[str] = None,
        model_id: Optional[str] = None
    ):
        self.dynamodb = boto3.resource('dynamodb')
        self.table_name = table_name or CLASSIFICATION_TABLE_NAME
        self.table = self.dynamodb.Table(self.table_name)
        
        self.bedrock = boto3.client('bedrock-runtime', region_name=AWS_REGION)
        self.model_id = model_id or CLASSIFICATION_MODEL_ID
        self.max_tokens = CLASSIFICATION_MAX_TOKENS
        self.temperature = CLASSIFICATION_TEMPERATURE
        
        logger.info(f"Initialized classifier: table={self.table_name}, model={self.model_id}")
    
    async def classify_document(
        self,
        filename: str,
        content_preview: str
    ) -> Dict[str, str]:
        """
        Classify document and return category info
        
        Returns:
            {
                'category_id': str,
                'vector_collection_name': str,
                'description': str,
                'is_new': bool
            }
        """
        # Get existing categories
        categories = await self._get_all_categories()
        
        # Classify using LLM
        classification = await self._classify_with_llm(filename, content_preview, categories)
        
        category_id = classification['category_id']
        
        # Check if category exists
        existing = await self._get_category(category_id)
        
        if not existing:
            # Create new category
            await self._create_category(classification)
            classification['is_new'] = True
        else:
            # Use existing
            classification['vector_collection_name'] = existing['vector_collection_name']
            classification['description'] = existing['description']
            classification['is_new'] = False
        
        # Increment document count
        await self._increment_count(category_id)
        
        return classification
    
    async def _get_all_categories(self) -> List[Dict]:
        """Get all existing categories from DynamoDB"""
        try:
            response = self.table.scan()
            items = response.get('Items', [])
            
            while 'LastEvaluatedKey' in response:
                response = self.table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
                items.extend(response.get('Items', []))
            
            return items
        except ClientError as e:
            logger.error(f"Error getting categories: {e}")
            return []
    
    async def _get_category(self, category_id: str) -> Optional[Dict]:
        """Get specific category"""
        try:
            response = self.table.get_item(Key={'category_id': category_id})
            return response.get('Item')
        except ClientError as e:
            logger.error(f"Error getting category {category_id}: {e}")
            return None
    
    async def _create_category(self, classification: Dict) -> bool:
        """Create new category in DynamoDB"""
        try:
            item = {
                'category_id': classification['category_id'],
                'vector_collection_name': classification['vector_collection_name'],
                'description': classification['description'],
                'keywords': classification.get('keywords', []),
                'document_count': 0
            }
            
            self.table.put_item(
                Item=item,
                ConditionExpression='attribute_not_exists(category_id)'
            )
            logger.info(f"Created category: {classification['category_id']}")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] != 'ConditionalCheckFailedException':
                logger.error(f"Error creating category: {e}")
            return False
    
    async def _increment_count(self, category_id: str) -> bool:
        """Increment document count"""
        try:
            self.table.update_item(
                Key={'category_id': category_id},
                UpdateExpression='ADD document_count :inc',
                ExpressionAttributeValues={':inc': 1}
            )
            return True
        except ClientError as e:
            logger.error(f"Error incrementing count: {e}")
            return False
    
    async def _classify_with_llm(
        self,
        filename: str,
        content_preview: str,
        existing_categories: List[Dict]
    ) -> Dict[str, str]:
        """Use Bedrock to classify document"""
        try:
            prompt = self._build_prompt(filename, content_preview, existing_categories)
            response_text = await self._call_bedrock(prompt)
            classification = self._parse_response(response_text)
            
            # Validate required fields
            required_fields = ['category_id', 'vector_collection_name', 'description']
            if not all(field in classification for field in required_fields):
                logger.warning(f"Missing required fields in classification response")
                return self._get_default_classification()
            
            return classification
            
        except Exception as e:
            logger.error(f"Classification error: {e}", exc_info=True)
            return self._get_default_classification()
    
    def _build_prompt(
        self,
        filename: str,
        content_preview: str,
        categories: List[Dict]
    ) -> str:
        """Build classification prompt"""
        categories_text = ""
        if categories:
            categories_text = "\n\nExisting Categories:\n"
            for cat in categories:
                keywords = ", ".join(cat.get('keywords', []))
                categories_text += f"- {cat.get('category_id')}: {cat.get('description')} [Keywords: {keywords}]\n"
        
        content_section = f"\n- First 5 lines:\n{content_preview}" if content_preview else ""
        
        return f"""You are a document classifier. Analyze the document carefully and classify it into the most specific and appropriate category.

Document Information:
- Filename: {filename}{content_section}
{categories_text}

Classification Rules:
1. **Read the filename and content carefully** to understand the document's specific purpose
2. **Prefer creating a NEW category** if the document has a distinct, specific purpose (e.g., policy booklet, claims form, user guide)
3. **Only reuse existing category** if it's an EXACT match in purpose (not just the same domain)
4. **Be specific**: "policy booklet" is different from "user info", "claims" is different from "policy", etc.

Domain Examples (create categories that match the specific document type):
   
Insurance:
- insurance_policy_booklet (policy terms, coverage details, T&Cs)
- insurance_user_info (customer personal data, contact info, demographics)
- insurance_vehicle_policy (vehicle-specific insurance, auto coverage)
- insurance_claims (claim forms, claim processing, claim reports)
- insurance_keycare_policy (roadside assistance, key replacement coverage)

Entertainment:
- entertainment_movies, entertainment_music, entertainment_events

Automobile:
- automobile_specifications, automobile_maintenance, automobile_sales

Corporate:
- company_policies, company_hr_docs, company_financial

Education:
- education_curriculum, education_research, education_assessments

Government:
- govt_policy, govt_regulations, govt_legal

General:
- general_documents, technical_manuals, legal_contracts

Format Requirements:
- Category ID: domain_specific_type (lowercase_with_underscores)
- Vector collection name: same as category_id
- Be specific in the category name to reflect the document type

Analyze the filename "{filename}" and content, then respond with JSON only:
{{{{
    "category_id": "insurance_policy_booklet",
    "vector_collection_name": "insurance_policy_booklet",
    "description": "Insurance policy booklets containing terms, conditions, and coverage details",
    "keywords": ["policy", "booklet", "terms", "coverage", "conditions"]
}}}}"""
    
    async def _call_bedrock(self, prompt: str) -> str:
        """Call Bedrock API"""
        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )

            print(request_body)
            print(response)
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        except Exception as e:
            logger.error(f"Bedrock error: {e}")
            raise
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse LLM JSON response"""
        try:
            # Find JSON block
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start == -1 or end == 0:
                logger.warning("No JSON found in response")
                return self._get_default_classification()
            
            json_str = response[start:end]
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['category_id', 'vector_collection_name', 'description']
            if all(field in data for field in required_fields):
                return data
            else:
                missing = [f for f in required_fields if f not in data]
                logger.warning(f"Missing fields in response: {missing}")
                return self._get_default_classification()
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            return self._get_default_classification()
        except Exception as e:
            logger.error(f"Parse error: {e}", exc_info=True)
            return self._get_default_classification()
    
    def _get_default_classification(self) -> Dict[str, str]:
        """Default category for unclassified docs"""
        return {
            "category_id": "general_documents",
            "vector_collection_name": "general_documents",
            "description": "General uncategorized documents",
            "keywords": ["general"]
        }
