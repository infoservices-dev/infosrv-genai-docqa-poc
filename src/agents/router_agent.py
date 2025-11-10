import os, json, boto3, logging, re
from decimal import Decimal

logger = logging.getLogger(__name__)
BEDROCK_RUNTIME = boto3.client("bedrock-runtime")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")


class BaseAgent:
    def __init__(self, config: dict):
        self.config = config

    def invoke(self, context: dict) -> dict:
        raise NotImplementedError()
    

def decimal_to_native(obj):
    """Convert DynamoDB Decimal types to native Python types"""
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    elif isinstance(obj, dict):
        return {k: decimal_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_native(v) for v in obj]
    return obj


class RoutingConfig:
    def __init__(self, table_name: str):
        self.table = boto3.resource("dynamodb").Table(table_name)

    def get_routing_map(self):
        try:
            items = self.table.scan().get("Items", [])
            routing_map = {}
            for i in items:
                routing_map[i["category_id"]] = decimal_to_native(
                    {
                        "collection_name": i.get("vector_collection_name"),
                        "description": i.get("description"),
                        "document_count": i.get("document_count", "N/A"),
                        "keywords": [
                            k.get("S") if isinstance(k, dict) else k for k in i.get("keywords", [])
                        ],
                        "domain": i.get("domain", "General"),
                    }
                )
            return routing_map
        except Exception as e:
            logger.error(f"Failed to fetch routing map: {e}")
            return {}


class LLMClassifier:
    def __init__(self, model_id):
        self.model_id = model_id

    def classify(self, query: str, routing_map: dict):
        prompt = self._build_prompt(routing_map)
        try:
            resp = BEDROCK_RUNTIME.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 150,
                        "system": prompt,
                        "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
                    }
                ),
            )
            body = json.loads(resp["body"].read())
            text = body["content"][0]["text"].strip()
            if "{" in text and "}" in text:
                return json.loads(text[text.find("{"):text.rfind("}") + 1]).get("category_id")
            return None
        except Exception as e:
            logger.error(f"Bedrock classify error: {e}")
            return None

    def _build_prompt(self, routing_map: dict):
        categories = "\n".join([f"- {k}: {v['description']}" for k, v in routing_map.items()])
        return (
            "You are a document router for internal knowledge base.\n"
            "Classify the user query strictly as JSON {\"category_id\": \"...\"} if it fits a category.\n"
            "If the question asks about available documents or freeform information, do not classify—return None.\n\n"
            f"Categories:\n{categories}"
        )


class RouterAgent(BaseAgent):
    GREETINGS = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]

    def __init__(self, table_name: str, model_id: str):
        self.routing_config = RoutingConfig(table_name)
        self.classifier = LLMClassifier(model_id)

    def invoke(self, event: dict) -> dict:
        query = event.get("query") or event.get("question")
        if not query:
            return {"status": "ERROR", "message": "Missing query."}

        routing_map = self.routing_config.get_routing_map()
        if not routing_map:
            return {"status": "ERROR", "message": "Routing config not found."}

        if query.strip().lower() in self.GREETINGS:
            return {
                "status": "CONTINUE",
                "next_agent": "SynthesisAgent",
                "query": query,
                "context_data": {"routing_map": routing_map},
            }

        category = self.classifier.classify(query, routing_map)

        if not category or category not in routing_map:
            return {
                "status": "CONTINUE",
                "next_agent": "SynthesisAgent",
                "query": query,
                "context_data": {"routing_map": routing_map, "classification_result": category},
            }

        config = routing_map[category]
        return {
            "status": "CONTINUE",
            "next_agent": "RetrievalAgent",
            "query": query,
            "context_data": {
                "routing_category": category,
                "search_scope": [config["collection_name"]],
                "domain": config["domain"],
                "category_config": config,
            },
        }


def lambda_handler(event, context):
    try:
        agent = RouterAgent(os.environ["ROUTING_TABLE_NAME"], MODEL_ID)
        return agent.invoke(event)
    except Exception as e:
        logger.error(f"RouterAgent failed: {e}")
        return {"status": "FATAL_ERROR", "message": str(e), "input": event}


if __name__ == "__main__":
    os.environ["ROUTING_TABLE_NAME"] = "genai-docqa-poc-dev-data-classification"
    test_query = "Hello, may i know key care policy details"
    #test_query = "may i know joes policy information"
    test_query = "what is policy about lost car key"
    print(lambda_handler({"query": test_query}, None))
