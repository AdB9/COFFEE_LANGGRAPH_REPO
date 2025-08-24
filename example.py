import logging
import time
import os
import asyncio
import ast
import json
import base64
from typing import Any, Dict, Optional, cast, Union, List, Sequence

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from hinge_agent.casa_nova.configuration import Configuration
from hinge_agent.casa_nova.schemas import ProfileAnalysis, AttributeWithConfidence
from hinge_agent.casa_nova.state import State

logger = logging.getLogger(__name__)
load_dotenv()


def _try_parse_custom_attributes(value: Any) -> Dict[str, Any]:
    """
    Safely parses a value that should be a dictionary, handling string-based
    outputs that are common from LLMs.
    """
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}

    logger.warning(
        f"Custom attributes returned as a string. Attempting to parse: '{value}'"
    )
    try:
        # ast.literal_eval is a safe way to evaluate a string containing a Python literal
        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        # Fallback for malformed strings that might be JSON-like (e.g., using single quotes)
        try:
            # A common LLM error is using single quotes; convert to double for JSON
            json_str = value.replace("'", '"')
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            logger.error(
                f"Failed to parse custom_attributes string as dict or JSON: {value}"
            )
            # Return an error marker in the dict for easier debugging
            return {"parsing_error": f"Could not decode string: {value}"}

    # If it's a string but not a dict literal (e.g., "N/A")
    return {"parsing_error": f"Value was a string but not a valid dict: {value}"}


def convert_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image


def create_human_message_with_images(
    image_paths: list[str], image_count: int
) -> HumanMessage:
    """Create a HumanMessage with multiple images properly formatted."""
    content: List[Union[str, Dict[str, Any]]] = [
        {
            "type": "text",
            "text": f"Please analyze these {image_count} dating profile images and extract only the information that is clearly visible and readable. Be conservative and accurate rather than comprehensive. If something is unclear or not visible, leave it empty.",
        }
    ]

    # Add each image to the content
    for image_path in image_paths:
        encoded_image = convert_image_to_base64(image_path)

        # Determine image format from file extension
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext == ".png":
            image_format = "png"
        elif file_ext in [".jpg", ".jpeg"]:
            image_format = "jpeg"
        else:
            image_format = "png"  # default to png

        content.append(
            {
                "type": "image_url",
                "image_url": f"data:image/{image_format};base64,{encoded_image}",
            }
        )

    return HumanMessage(content=content)


def load_attribute_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load attribute configuration from JSON file."""
    if config_path is None:
        # Default to the example config in the same directory as schemas
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "attribute_config_example.json"
        )

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(
            f"Attribute config file not found at {config_path}, using minimal defaults"
        )
        return {
            "physical_attributes": [],
            "profile_attributes": [],
            "user_preferences": {},
        }


def create_extraction_system_message(
    custom_prompt: Optional[str] = None,
    config_path: Optional[str] = None,
    for_structured_output: bool = True,
) -> str:
    """Create the system message for profile analysis with configurable attribute extraction."""

    # Load attribute configuration
    attr_config = load_attribute_config(config_path)

    # Build physical attributes instructions
    physical_attrs_instructions = ""
    if attr_config.get("physical_attributes"):
        physical_attrs_instructions = "\n   PHYSICAL ATTRIBUTES (extract as key-value pairs with confidence scores):\n"
        for attr in attr_config["physical_attributes"]:
            physical_attrs_instructions += (
                f"   - {attr['key']}: {attr['description']}\n"
            )

    # Build profile attributes instructions
    profile_attrs_instructions = ""
    if attr_config.get("profile_attributes"):
        profile_attrs_instructions = "\n   PROFILE ATTRIBUTES (extract as key-value pairs with confidence scores):\n"
        for attr in attr_config["profile_attributes"]:
            profile_attrs_instructions += f"   - {attr['key']}: {attr['description']}\n"

    base_extraction_instructions = f"""
    - age: Estimated or stated age - look for age numbers, birth years, or visual estimates
    - location: City or location mentioned - check for location tags, city names
    - occupation: Job or profession mentioned - look for job titles, workplace mentions, career indicators
    - education: Educational background if mentioned - check for educational institutions, graduation years
    - relationship_goals: What they're looking for in relationships
    - interests: Hobbies, interests, or activities mentioned - extract from text, photos, and visual cues
    - lifestyle: Lifestyle indicators (drinking, smoking, fitness habits, social activities)
    - dealbreakers: Any potential red flags or concerning elements
    - positive_indicators: Positive traits or attractive qualities
    {physical_attrs_instructions}{profile_attrs_instructions}
    
    NOTE: Only extract information that is clearly visible. Do not include fields for name or verification status as they are not part of the schema.
    """

    custom_instructions = ""
    if custom_prompt:
        custom_instructions = (
            """
3. CUSTOM ATTRIBUTES:
   Extract any information related to the following custom topics:
   """
            + custom_prompt
            + """
   
   Place your findings into the `custom_attributes` field. This field **MUST** be a dictionary (a JSON object) with clear key-value pairs. For example:
   "custom_attributes": {
     "lifestyle_cues": ["Enjoys outdoor activities", "Social person", "Likes fine dining"],
     "personality_traits": ["Adventurous", "Outgoing", "Career-focused"]
   }
   
   IMPORTANT: The `custom_attributes` field cannot be a plain string; it must be a structured JSON object."""
        )

    # Different output format instructions based on whether we're using structured output
    if for_structured_output:
        output_format_instructions = """
IMPORTANT: You are using structured output. Do NOT format your response as JSON or use markdown code blocks. 
Simply provide the data directly according to the ProfileAnalysis schema structure:

- extracted_text: object with prompt_responses array and other_text array
- profile_attributes: object with all the profile fields
- image_count: number of images analyzed
- confidence_score: your confidence in the analysis (0.0-1.0)
- analysis_notes: string with your observations

Provide the raw data directly - no JSON formatting, no code blocks, no markdown."""
    else:
        output_format_instructions = """
OUTPUT FORMAT:
- Use "extracted_text" with "prompt_responses" array of objects containing prompt-response pairs
- Each prompt_response object should have "prompt" and "response" fields
- Use "other_text" array for any other clearly readable text
- Use "profile_attributes" for personal details you can clearly extract
- For configurable attributes, use the "attributes" array with key-value-confidence objects
- For physical attributes, use the nested "physical_attributes.attributes" array
- Include confidence scores (0.0-1.0) for each attribute based on how certain you are
- Include "analysis_notes" with factual observations about what you can see
- For custom_attributes, use a dictionary/object format, not a string
- Leave fields empty/null if information is not clearly visible

Your final output must be a single, valid JSON object that strictly adheres to the ProfileAnalysis schema. Only include information you can clearly and confidently extract from the images."""

    return f"""You are an expert at analyzing dating profile images to extract information accurately. ACCURACY IS MORE IMPORTANT THAN COMPLETENESS.

Your task is to analyze the provided images and extract:

1. TEXTUAL CONTENT - ONLY WHAT IS CLEARLY VISIBLE:
   - Extract only prompts/questions that are clearly visible and readable in the images
   - Extract only the corresponding answers/responses that are clearly visible
   - Capture other text only if it's clearly readable (age numbers, location names, etc.)
   - If text is blurry, partially obscured, or unclear, do NOT attempt to guess what it says

2. VISUAL ANALYSIS - BE CONSERVATIVE:
   - Describe only what you can clearly see in each photo
   - Only make reasonable inferences based on clear visual evidence
   - Avoid detailed speculation about personality or lifestyle without clear evidence
   - If a photo is unclear or ambiguous, state that rather than making assumptions

3. PROFILE ATTRIBUTES - EXTRACT CONSERVATIVELY:
{base_extraction_instructions}
{custom_instructions}

CRITICAL ACCURACY RULES:
- ONLY extract information that is clearly visible or directly readable
- DO NOT fill in fields with guessed or assumed information
- If you cannot clearly see something, leave that field empty/null
- Better to have empty fields than incorrect information
- For personality traits, only include those clearly supported by visible evidence
- For analysis_notes, describe only what you can actually see, not what you imagine

IMPORTANT: If information is not clearly visible in the images, it's better to leave fields empty than to guess. Accuracy trumps completeness.

REQUIRED OUTPUT STRUCTURE (use these exact field names):
{{
  "extracted_text": {{
    "prompt_responses": [
      {{"prompt": "clearly visible prompt", "response": "clearly visible response"}},
      {{"prompt": "another prompt", "response": "another response"}}
    ],
    "other_text": ["other clearly readable text"]
  }},
  "profile_attributes": {{
    "age": null_or_number,
    "location": null_or_string,
    "occupation": null_or_string,
    "education": null_or_string,
    "relationship_goals": null_or_string,
    "attributes": [
      {{"key": "attribute_name", "value": "attribute_value", "confidence": 0.0_to_1.0}},
      {{"key": "another_attribute", "value": "another_value", "confidence": 0.0_to_1.0}}
    ],
    "physical_attributes": {{
      "height": null_or_string,
      "attributes": [
        {{"key": "physical_attribute_name", "value": "attribute_value", "confidence": 0.0_to_1.0}},
        {{"key": "another_physical_attribute", "value": "another_value", "confidence": 0.0_to_1.0}}
      ],
      "physical_activity_indicators": ["list of activity evidence"],
      "dress_style": ["list of clothing styles observed"],
      "photo_quality_factors": ["factors affecting assessment quality"]
    }},
    "interests": ["list of clear interests"],
    "lifestyle": ["clear lifestyle indicators"],
    "dealbreakers": ["clear red flags only"],
    "positive_indicators": ["clear positive traits"],
    "overall_appeal_score": 0.0_to_1.0,
    "custom_attributes": {{dictionary_format}}
  }},
  "image_count": number,
  "confidence_score": 0.0_to_1.0,
  "analysis_notes": "factual description of what you can see"
}}

{output_format_instructions}"""


def _apply_schema_defaults(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply default values to match the ProfileAnalysis schema expectations."""
    # Ensure extracted_text exists and has proper defaults
    if "extracted_text" not in analysis_dict:
        analysis_dict["extracted_text"] = {}

    extracted_text = analysis_dict["extracted_text"]
    extracted_text.setdefault("prompt_responses", [])
    extracted_text.setdefault("other_text", [])

    # Ensure profile_attributes exists and has proper defaults
    if "profile_attributes" not in analysis_dict:
        analysis_dict["profile_attributes"] = {}

    profile_attrs = analysis_dict["profile_attributes"]

    # Set list fields to empty lists if they are None
    list_fields = ["interests", "lifestyle", "dealbreakers", "positive_indicators"]
    for field in list_fields:
        if profile_attrs.get(field) is None:
            profile_attrs[field] = []

    # Ensure attributes list exists
    if "attributes" not in profile_attrs:
        profile_attrs["attributes"] = []

    # Ensure physical_attributes exists and has proper structure
    if "physical_attributes" not in profile_attrs:
        profile_attrs["physical_attributes"] = {}

    phys_attrs = profile_attrs["physical_attributes"]
    if "attributes" not in phys_attrs:
        phys_attrs["attributes"] = []
    phys_attrs.setdefault("physical_activity_indicators", [])
    phys_attrs.setdefault("dress_style", [])
    phys_attrs.setdefault("photo_quality_factors", [])

    # Set custom_attributes to empty dict if it's None
    if profile_attrs.get("custom_attributes") is None:
        profile_attrs["custom_attributes"] = {}

    # Set other default values
    analysis_dict.setdefault("image_count", 0)
    analysis_dict.setdefault("confidence_score", 0.0)
    analysis_dict.setdefault("analysis_notes", "")

    return analysis_dict


async def analyze_profile(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Analyze profile images to extract text and attributes."""
    start_time = time.time()
    logger.info("---PROFILE ANALYZER AGENT---")

    try:
        state.visited_agents.add("profile_analyzer")
        state.current_agent = "profile_analyzer"

        if not state.images:
            return {"error": "No images provided for analysis."}

        logger.info(f"Analyzing {len(state.images)} profile images")

        configuration = Configuration.from_runnable_config(config)

        google_api_key = os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model=configuration.model.replace("google_genai:", ""),
            temperature=configuration.temperature,
            max_tokens=configuration.max_tokens,
            google_api_key=google_api_key,
        )

        # Create the human message with images
        human_message = create_human_message_with_images(
            state.images, len(state.images)
        )

        # Create system message with configurable attributes
        system_message = create_extraction_system_message(
            state.custom_extraction_prompt, for_structured_output=True
        )

        # Create messages list
        messages = [("system", system_message), human_message]

        # Try structured output first
        try:
            logger.info("Invoking profile analysis with structured output...")
            analysis_result = await llm.with_structured_output(ProfileAnalysis).ainvoke(
                messages, config
            )

            if analysis_result is None:
                logger.error(
                    "Structured output returned None - this indicates an issue with the LLM call or prompt"
                )
                # Let's try a simple raw call to see what the LLM is actually returning
                logger.info("Attempting raw LLM call to diagnose the issue...")
                raw_result = await llm.ainvoke(messages, config)
                logger.info(f"Raw LLM result type: {type(raw_result)}")
                logger.info(
                    f"Raw LLM content (first 2000 chars): {str(raw_result.content if hasattr(raw_result, 'content') else raw_result)[:2000]}"
                )
                raise ValueError("Structured output returned None")

            logger.info("Structured output succeeded!")

        except Exception as e:
            logger.error(f"Structured output failed: {e}")
            raise e

        # The LLM can sometimes return a dict or a Pydantic object with errors.
        # This block ensures the final object is a valid ProfileAnalysis instance.
        if analysis_result is None:
            logger.error("Analysis result is None - creating minimal analysis")
            # Create a minimal analysis with default values
            analysis_result = {
                "extracted_text": {"prompt_responses": [], "other_text": []},
                "profile_attributes": {
                    "age": None,
                    "location": None,
                    "occupation": None,
                    "education": None,
                    "relationship_goals": None,
                    "attributes": [],
                    "physical_attributes": {
                        "height": None,
                        "attributes": [],
                        "physical_activity_indicators": [],
                        "dress_style": [],
                        "photo_quality_factors": [],
                    },
                    "interests": [],
                    "lifestyle": [],
                    "dealbreakers": [],
                    "positive_indicators": [],
                    "overall_appeal_score": 0.0,
                    "custom_attributes": {},
                },
                "image_count": len(state.images),
                "confidence_score": 0.1,
                "analysis_notes": "Analysis failed - LLM returned no data",
            }

        if isinstance(analysis_result, dict):
            # Apply schema defaults before validation
            analysis_result = _apply_schema_defaults(analysis_result)

            # If the LLM returns a raw dict, repair custom_attributes and analysis_notes before Pydantic validation.
            if "profile_attributes" in analysis_result and isinstance(
                analysis_result["profile_attributes"], dict
            ):
                attrs = analysis_result["profile_attributes"]
                if "custom_attributes" in attrs:
                    attrs["custom_attributes"] = _try_parse_custom_attributes(
                        attrs.get("custom_attributes")
                    )

            # Fix analysis_notes if it's a list instead of a string
            if "analysis_notes" in analysis_result and isinstance(
                analysis_result["analysis_notes"], list
            ):
                # Convert list of notes to a single string
                notes_list = analysis_result["analysis_notes"]
                if notes_list:
                    # Join the notes into a formatted string
                    analysis_result["analysis_notes"] = "\n".join(
                        [
                            f"Image {note.get('image_number', i + 1)}: {note.get('description', str(note))}"
                            if isinstance(note, dict)
                            else str(note)
                            for i, note in enumerate(notes_list)
                        ]
                    )
                else:
                    analysis_result["analysis_notes"] = ""

            profile_analysis = ProfileAnalysis(**analysis_result)
        elif isinstance(analysis_result, ProfileAnalysis):
            profile_analysis = analysis_result
            # If already a Pydantic object, ensure nested custom_attributes are correctly typed.
            if profile_analysis.profile_attributes:
                profile_analysis.profile_attributes.custom_attributes = (
                    _try_parse_custom_attributes(
                        profile_analysis.profile_attributes.custom_attributes
                    )
                )
        else:
            logger.error(f"LLM returned unexpected type: {type(analysis_result)}")
            # Create a fallback analysis instead of raising an error
            logger.warning("Creating fallback analysis due to unexpected return type")
            # Create a dictionary that will be processed like any other LLM output
            analysis_result = {
                "extracted_text": {"prompt_responses": [], "other_text": []},
                "profile_attributes": {
                    "age": None,
                    "location": None,
                    "occupation": None,
                    "education": None,
                    "relationship_goals": None,
                    "attributes": [],
                    "physical_attributes": {
                        "height": None,
                        "attributes": [],
                        "physical_activity_indicators": [],
                        "dress_style": [],
                        "photo_quality_factors": [],
                    },
                    "interests": [],
                    "lifestyle": [],
                    "dealbreakers": [],
                    "positive_indicators": [],
                    "overall_appeal_score": 0.0,
                    "custom_attributes": {},
                },
                "image_count": len(state.images),
                "confidence_score": 0.1,
                "analysis_notes": f"Analysis failed - LLM returned unexpected type: {type(analysis_result)}",
            }
            # Apply schema defaults and create ProfileAnalysis like we do for dictionaries
            analysis_result = _apply_schema_defaults(analysis_result)
            profile_analysis = ProfileAnalysis(**analysis_result)

        # Calculate a dynamic confidence score if the LLM didn't provide one.
        if profile_analysis.confidence_score == 0.0:
            # Base confidence on quality indicators rather than quantity
            has_clear_text = (
                profile_analysis.extracted_text
                and len(profile_analysis.extracted_text.prompt_responses) > 0
            )
            has_basic_attributes = False
            if profile_analysis.profile_attributes:
                has_basic_attributes = any(
                    [
                        profile_analysis.profile_attributes.age,
                        profile_analysis.profile_attributes.location,
                        profile_analysis.profile_attributes.occupation,
                    ]
                )

            # Conservative scoring - start low and only increase with clear evidence
            if has_clear_text and has_basic_attributes:
                profile_analysis.confidence_score = 0.7
            elif has_clear_text or has_basic_attributes:
                profile_analysis.confidence_score = 0.5
            else:
                profile_analysis.confidence_score = 0.3

        profile_analysis.image_count = len(state.images)
        processing_time = time.time() - start_time
        logger.info(f"Profile analysis completed in {processing_time:.2f}s")
        logger.info(f"Confidence score: {profile_analysis.confidence_score:.2f}")

        return {"profile_analysis": profile_analysis}

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            f"Profile analysis failed after {processing_time:.2f}s: {e}", exc_info=True
        )
        return {
            "error": f"Profile analysis failed: {str(e)}",
            "final_answer": "I apologize, but an error occurred while analyzing the profile images.",
        }


def get_test_images():
    """Get test images for individual testing."""
    images = []

    test_dir = "/Users/tilaksharma/VsCodeProjects/hinge_agent/profile_2"
    if os.path.exists(test_dir):
        for filename in sorted(os.listdir(test_dir)):
            if filename.endswith(".PNG"):
                images.append(os.path.join(test_dir, filename))
    return images


async def main():
    """Test the profile analyzer individually."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info("🔍 Testing Profile Analyzer Agent Individually")
    test_images = get_test_images()
    if not test_images:
        logger.error("No test images found!")
        return
    logger.info(f"Found {len(test_images)} test images")
    config = cast(
        RunnableConfig,
        {
            "configurable": {
                "model": "gemini-2.5-flash",
                "temperature": 0.5,
                "max_tokens": 4096,
            }
        },
    )
    custom_prompt = "Pet preferences, travel interests, fitness routine, music taste, and political views."
    state = State(images=test_images, custom_extraction_prompt=custom_prompt)

    try:
        result = await analyze_profile(state, config)
        if result.get("error"):
            logger.error(f"❌ Error: {result['error']}")
            return
        if result.get("profile_analysis"):
            analysis = result["profile_analysis"]
            logger.info("\n📊 Profile Analysis Results:")
            logger.info(f"   Images analyzed: {analysis.image_count}")
            logger.info(f"   Confidence: {analysis.confidence_score:.1%}")

            if analysis.extracted_text:
                logger.info(f"\n📝 Extracted Text:")
                for p in analysis.extracted_text.prompt_responses:
                    logger.info(f"   [Prompt] {p.prompt}")
                    logger.info(f"   [Response] {p.response}")
                for o in analysis.extracted_text.other_text:
                    logger.info(f"   [Other Text] {o}")

            if analysis.profile_attributes:
                logger.info(f"\n👤 Profile Attributes:")
                attrs = analysis.profile_attributes

                # Log basic demographic fields
                basic_fields = [
                    "age",
                    "location",
                    "occupation",
                    "education",
                    "relationship_goals",
                ]
                for field in basic_fields:
                    value = getattr(attrs, field, None)
                    if value:
                        logger.info(f"   - {field.replace('_', ' ').title()}: {value}")

                # Log configurable attributes
                if attrs.attributes:
                    logger.info(f"   - Profile Attributes:")
                    for attr in attrs.attributes:
                        logger.info(
                            f"     • {attr.key}: {attr.value} (confidence: {attr.confidence:.2f})"
                        )

                # Log physical attributes
                if attrs.physical_attributes:
                    phys_attrs = attrs.physical_attributes
                    if phys_attrs.height:
                        logger.info(f"   - Height: {phys_attrs.height}")
                    if phys_attrs.attributes:
                        logger.info(f"   - Physical Attributes:")
                        for attr in phys_attrs.attributes:
                            logger.info(
                                f"     • {attr.key}: {attr.value} (confidence: {attr.confidence:.2f})"
                            )
                    if phys_attrs.physical_activity_indicators:
                        logger.info(
                            f"   - Activity Indicators: {phys_attrs.physical_activity_indicators}"
                        )
                    if phys_attrs.dress_style:
                        logger.info(f"   - Dress Style: {phys_attrs.dress_style}")

                # Log list fields
                list_fields = [
                    "interests",
                    "lifestyle",
                    "dealbreakers",
                    "positive_indicators",
                ]
                for field in list_fields:
                    value = getattr(attrs, field, None)
                    if value:
                        logger.info(f"   - {field.replace('_', ' ').title()}: {value}")

                # Log overall appeal score
                if attrs.overall_appeal_score:
                    logger.info(
                        f"   - Overall Appeal Score: {attrs.overall_appeal_score:.2f}"
                    )

                # Log custom attributes
                if attrs.custom_attributes:
                    logger.info(f"   - Custom Attributes: {attrs.custom_attributes}")

            logger.info("\n✅ Profile analyzer test completed successfully!")
    except Exception as e:
        logger.error(f"❌ Profile analyzer test failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
