import os
import json
import boto3

BEDROCK_RUNTIME = boto3.client("bedrock-runtime")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

class BaseAgent:
    def __init__(self, config: dict):
        self.config = config

    def invoke(self, context: dict) -> dict:
        raise NotImplementedError("Invoke must be implemented by subclasses.")

class SynthesisAgent(BaseAgent):
    def invoke(self, context: dict) -> dict:
        query = context.get("query")
        if not query:
            return {"status": "ERROR", "message": "Missing query in context."}

        # Handle different input structures from Step Functions
        router_result = context.get("routerResult", {})
        retrieval_result = context.get("retrievalResult", {})
        
        # Get context data from the appropriate source
        if retrieval_result and retrieval_result.get("context_data"):
            # Coming from RetrievalAgent
            context_data = self._load_context(retrieval_result.get("context_data"))
        else:
            # Coming directly from RouterAgent
            context_data = self._load_context(router_result.get("context_data")) \
                          or self._load_context(context.get("context_data"))

        if context_data.get("retrieved_texts"):
            return self._handle_retrieval_response(query, context_data)
        else:
            return self._handle_general_response(query, context_data)

    @staticmethod
    def _load_context(raw_data):
        if not raw_data:
            return {}
        if isinstance(raw_data, dict):
            return raw_data
        try:
            return json.loads(raw_data)
        except Exception:
            return {}

    def _handle_retrieval_response(self, query: str, context_data: dict) -> dict:
        retrieved_texts = context_data.get("retrieved_texts", [])
        source_metadata = context_data.get("source_metadata", [])
        routing_category = context_data.get("routing_category")

        if not retrieved_texts:
            return {
                "status": "COMPLETED",
                "final_answer": f"No relevant documents found for '{query}'.",
                "citations": []
            }

        prompt = self._generate_retrieval_prompt(query, retrieved_texts, source_metadata)
        llm_result = self._invoke_bedrock_llm(prompt, source_metadata)

        return {
            "status": "COMPLETED",
            "query": query,
            "routing_category": routing_category,
            **llm_result
        }

    def _handle_general_response(self, query: str, context_data: dict) -> dict:
        routing_map = context_data.get("routing_map", {})
        classification_result = context_data.get("classification_result")

        if self._is_greeting(query):
            return self._generate_greeting_response(routing_map)
        elif classification_result is None:
            return self._generate_summary_response(query, routing_map)
        else:
            return self._generate_fallback_response(query)

    @staticmethod
    def _is_greeting(query: str) -> bool:
        return query.strip().lower() in ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]

    @staticmethod
    def _generate_greeting_response(routing_map: dict) -> dict:
        categories = list(routing_map.keys())
        category_descriptions = [f"• {cat}: {info['description']}" for cat, info in routing_map.items()]
        response = (
            "Hello! I'm your document assistant. I can help you find information from our knowledge base.\n\n"
            f"Available document categories:\n{chr(10).join(category_descriptions)}\n\nHow can I assist you today?"
        )
        return {"status": "COMPLETED", "final_answer": response, "categories": categories, "citations": []}

    def _generate_summary_response(self, query: str, routing_map: dict) -> dict:
        prompt = self._build_summary_prompt(query, routing_map)
        try:
            resp = BEDROCK_RUNTIME.invoke_model(
                modelId=MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "temperature": 0.3,
                    "system": prompt,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}]
                })
            )
            response_body = json.loads(resp.get("body").read())
            answer = response_body["content"][0]["text"].strip()
            if not answer:
                return self._generate_fallback_response(query)
            return {
                "status": "COMPLETED",
                "final_answer": answer,
                "query_type": "general_inquiry",
                "categories": list(routing_map.keys()),
                "citations": []
            }
        except Exception as e:
            print(f"Error in summary response: {e}")
            return self._generate_fallback_response(query)

    @staticmethod
    def _generate_fallback_response(query: str) -> dict:
        categories = [
            "insurance_policy_booklet", "insurance_vehicle_policy", "insurance_keycare_policy",
            "insurance_credit_agreement", "insurance_claims"
        ]
        response_text = (
            f"I understand you're asking about '{query}'. I have access to several insurance document categories:\n"
            "• Insurance Policy Booklets - Terms, conditions, and coverage details\n"
            "• Vehicle Insurance Policies - Auto coverage and claims information\n"
            "• KeyCare Insurance Policies - Key replacement and locksmith services\n"
            "• Credit Agreements - Premium financing and payment terms\n"
            "• Insurance Claims - Filing procedures and claim processing\n\n"
            "Could you be more specific about which type of insurance information you're looking for?"
        )
        return {"status": "COMPLETED", "final_answer": response_text, "categories": categories, "citations": []}

    @staticmethod
    def _build_summary_prompt(query: str, routing_map: dict) -> str:
        categories_info = "\n".join(
            [f"- {cat}: {info['description']} (Documents: {info.get('document_count', 'N/A')})"
             for cat, info in routing_map.items()]
        )
        return (
            f"You are a helpful document assistant for an insurance knowledge base. User query: '{query}'\n"
            f"Available document categories:\n{categories_info}\n"
            "Instructions:\n"
            "1. Provide a helpful, specific response.\n"
            "2. Suggest relevant categories.\n"
            "3. Keep responses concise (2-3 paragraphs max).\n"
            "4. Always offer to help with more specific questions."
        )

    @staticmethod
    def _generate_retrieval_prompt(query: str, retrieved_context: list, source_metadata: list) -> str:
        context_blocks = [f"Document Chunk [{i+1}]:\n{chunk}" for i, chunk in enumerate(retrieved_context)]
        citations_list = [f"[{i+1}]: {meta['source_uri']} (Chunk ID: {meta['chunk_id']})"
                          for i, meta in enumerate(source_metadata)]
        return (
            "<system>You are an expert Q&A system. Answer using only the provided document chunks.\n"
            f"{chr(10).join(context_blocks)}\nCitations:\n{chr(10).join(citations_list)}\n</system>\n"
            f"<user_query>{query}</user_query>"
        )

    @staticmethod
    def _invoke_bedrock_llm(prompt: str, source_metadata: list) -> dict:
        try:
            system_prompt, user_content = prompt.split("</system>")
            system_prompt = system_prompt.replace("<system>", "").strip()
            user_content = user_content.strip()
            response = BEDROCK_RUNTIME.invoke_model(
                modelId=MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2048,
                    "temperature": 0.1,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": user_content}]}]
                })
            )
            body = json.loads(response.get("body").read())
            text = body["content"][0]["text"].strip()
            
            # Try to extract JSON if present, otherwise use the raw text
            try:
                json_start = text.find("{")
                json_end = text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    llm_result = json.loads(text[json_start:json_end])
                    return {"final_answer": llm_result.get("answer", text), "citations": source_metadata}
                else:
                    # No JSON found, return the raw text as the answer
                    return {"final_answer": text, "citations": source_metadata}
            except json.JSONDecodeError:
                # JSON parsing failed, return the raw text
                return {"final_answer": text, "citations": source_metadata}
                
        except Exception as e:
            print(f"Bedrock synthesis failed: {e}")
            return {"final_answer": "Answer generation failed.", "citations": []}


def lambda_handler(event, context):
    agent = SynthesisAgent(config={})
    try:
        payload = event.get("event", event)
        if "query" not in payload:
            return {"status": "ERROR", "error_message": "Missing 'query'", "input_context": payload}

        return agent.invoke(payload)
    except Exception as e:
        print(f"Synthesis Lambda failed: {e}")
        return {"status": "FATAL_ERROR", "error_message": str(e), "input_context": event}


if __name__ == "__main__":
    os.environ["BEDROCK_MODEL_ID"] = "anthropic.claude-3-sonnet-20240229-v1:0"
    os.environ["ROUTING_TABLE_NAME"] = "genai-docqa-poc-dev-data-classification"
    test_event = {'status': 'CONTINUE', 'next_agent': 'SynthesisAgent', 'query': 'hi', 'context_data': '{"routing_map": {"insurance_policy_booklet": {"collection_name": "insurance_policy_booklet", "description": "Insurance policy booklets containing terms, conditions, and coverage details", "document_count": 1, "keywords": ["policy", "booklet", "terms", "coverage", "conditions"], "domain": "General"}, "insurance_vehicle_policy": {"collection_name": "insurance_vehicle_policy", "description": "Documents related to vehicle-specific insurance policies, including auto coverage details, terms, and conditions", "document_count": 4, "keywords": ["car insurance", "vehicle policy", "auto coverage", "terms", "conditions"], "domain": "General"}, "insurance_keycare_policy": {"collection_name": "insurance_keycare_policy", "description": "Documents related to key insurance policies, including coverage for lost/stolen keys, key replacement, locksmith services, and related terms and conditions", "document_count": 1, "keywords": ["key insurance", "lost keys", "stolen keys", "key replacement", "locksmith", "terms", "conditions"], "domain": "General"}, "insurance_credit_agreement": {"collection_name": "insurance_credit_agreement", "description": "Documents related to credit agreements for financing insurance premiums, including terms, conditions, and explanations", "document_count": 2, "keywords": ["credit agreement", "financing", "premiums", "terms", "conditions", "explanation"], "domain": "General"}, "insurance_claims": {"collection_name": "insurance_claims", "description": "Documents related to filing insurance claims, claim processing, and claim reports", "document_count": 3, "keywords": ["claims", "accident", "report", "processing", "advisers"], "domain": "General"}}}'}
    test_event = {'status': 'CONTINUE', 'next_agent': 'SynthesisAgent', 'query': 'Which type key policy are covered through my policy', 'context_data': '{"routing_category": "insurance_keycare_policy", "retrieved_texts": [], "source_metadata": [], "category_config": {"collection_name": "insurance_keycare_policy", "description": "Insurance policy booklets containing terms, conditions, and coverage details", "document_count": 1, "keywords": ["policy", "booklet", "terms", "coverage", "conditions"], "domain": "General"}, "search_scope": ["insurance_keycare_policy"]}'}
    test_event ={'status': 'CONTINUE', 'next_agent': 'SynthesisAgent', 'query': 'what is policy about lost car key', 'context_data': {'routing_category': 'insurance_keycare_policy', 'retrieved_texts': ['key crime. • Thieves are increasingly trying new methods of vehicle crime. This means stealing your keys to your vehicle first. Burglars have been known to break into houses and offices just to steal vehicle keys. • Don’t leave vehicle keys close to the front door where they can be seen. • NEVER leave your keys in your vehicle - not even for a second. This is especially important when at a petrol station or when loading or unloading your vehicle. • Always lock your vehicle when leaving it. \n  Terms and Cond', 'y and includes any reprogramming of infrared handsets, immobilisers and alarms necessitated by such replacement of the Insured Key. Security risk: The risk resulting from the accidental loss of an Insured Key where it is possible for someone who found the key to trace it to Your vehicle or premises. Statement of Facts: The statement produced by Keycare following authorisation of a claim. Territorial limits: Worldwide. Vehicle hire charges: The standard charges (excluding any optional extras) up to a maximum', 'e £10 reward? A No. The reward will be sent directly by Keycare Limited to the person who found your keys.    Additional fobs are available to protect your additional sets of keys and keys for your family members residing at the same address as you. Each additional fob provides up to the maximum cover limit, as detailed in your Policy Schedule.  Sophisticated security measures now fitted as standard to new vehicles mean criminals are increasingly turning to key crime. • Thieves are increasingly trying new m', 'es (excluding any optional extras) up to a maximum of £50 a day to hire a vehicle for a period of up to three days. Waiting Period: A period of 48 hours commencing when the loss of the Insured Key is first reported to Keycare. Wear and Tear: The gradual loss of an Insured Key’s ability to function exactly as it was designed to do by the manufacturer due solely to the passage of time and repeated usage. You/Your: The Policyholder, any Immediate Member of the Policyholder’s family permanently living with the', 'AnyCompany Car Insurance Keycare Policy Booklet 07777 7777                         \nMake a note of your unique fob number here.   If your keys go missing call our emergency helpline number immediately on  We are ready to take your call.  Welcome to Keycare The leading name in the recovery and replacement of lost or stolen keys This is your policy booklet. It sets out the details of your policy and should be read in conjunction with your Keycare Policy Schedule. Please keep these documents safe.  6 step proc'], 'source_metadata': [{'source_uri': 'keycare_policy_booklet.pdf', 'chunk_id': 8, 'score': 0.6621190299999999}, {'source_uri': 'keycare_policy_booklet.pdf', 'chunk_id': 21, 'score': 0.6517303}, {'source_uri': 'keycare_policy_booklet.pdf', 'chunk_id': 7, 'score': 0.64340496}, {'source_uri': 'keycare_policy_booklet.pdf', 'chunk_id': 22, 'score': 0.6231525}, {'source_uri': 'keycare_policy_booklet.pdf', 'chunk_id': 'e3cdf7a1-f228-4fca-9575-19c3de36ae77', 'score': 0.6162547}], 'category_config': {'collection_name': 'insurance_keycare_policy', 'description': 'Documents related to key insurance policies, including coverage for lost/stolen keys, key replacement, locksmith services, and related terms and conditions', 'document_count': 1, 'keywords': ['key insurance', 'lost keys', 'stolen keys', 'key replacement', 'locksmith', 'terms', 'conditions'], 'domain': 'General'}, 'search_scope': ['insurance_keycare_policy']}}
    print(lambda_handler(test_event, None))