import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from the parent workspace .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))

# Fallback environment check
if not os.environ.get("GEMINI_API_KEY"):
    # Attempt to load from current directory .env if any
    load_dotenv()

class AstraCopilot:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            # Clean up key if it contains comments or placeholders
            api_key = api_key.split(" ")[0].strip()
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.has_api = True
        else:
            self.has_api = False
            print("Warning: GEMINI_API_KEY not found. Copilot will run in mock mode.")

    def chat_with_copilot(self, query: str, network_context: dict) -> str:
        """
        Sends the user query along with real-time prediction and optimization context to Gemini.
        """
        system_prompt = (
            "You are AstraTraffic AI Copilot, a senior decision-support agent for traffic command rooms.\n"
            "Your job is to translate complex machine learning forecasts, optimization models, and simulation "
            "metrics into concrete, tactical, and clear operational briefings for traffic police and city authorities.\n"
            "Format your response in neat, professional Markdown. Be direct, authoritative, and precise.\n"
        )
        
        context_str = f"""
### CURRENT TRAFFIC NETWORK CONTEXT:
- Active Incidents monitored: {network_context.get('incident_count', 0)}
- Global Network Congestion Level (0-10): {network_context.get('average_congestion', 3.5)}
- Key Bottlenecks Identified: {', '.join(network_context.get('bottlenecks', ['None']))}
- Active Manpower Roster Count: {network_context.get('available_officers', 20)} Officers deployed

### MODEL FORECASTS:
- Expected duration of new incident: {network_context.get('predicted_duration_mins', 'N/A')} mins
- Road closure probability: {network_context.get('road_closure_probability', 0.0) * 100}%
- Priority Level: {network_context.get('priority', 'Low')}

### RECOMMENDATION ENGINE SUMMARY:
- Police Dispatch suggestion: {network_context.get('manpower_recommendation', 'None')}
- Signal timings offset action: {network_context.get('signal_strategy', 'None')}
- Barricading instruction: {network_context.get('barricade_strategy', 'None')}
"""

        user_content = f"{context_str}\n\n### USER QUERY:\n{query}\n\n### RESPONSE:"
        
        if not self.has_api:
            # Mock responses if API key is not active/valid
            return self._mock_response(query, network_context)
            
        try:
            response = self.model.generate_content(
                contents=[
                    {"role": "user", "parts": [system_prompt + "\n" + user_content]}
                ]
            )
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return self._mock_response(query, network_context)

    def _mock_response(self, query: str, context: dict) -> str:
        # High quality fallback responses for common queries if API is offline
        q = query.lower()
        if 'manpower' in q or 'police' in q or 'officer' in q:
            return (
                "### 🚨 Manpower Allocation Briefing\n\n"
                f"Based on current incidents, we have **{context.get('available_officers', 20)} officers** in rotation. "
                "The optimization engine recommends the following prioritization:\n"
                "- **Primary Dispatch**: Deploy 3 officers to the highest-severity bottleneck (predicted duration: "
                f"{context.get('predicted_duration_mins', 64)} minutes).\n"
                "- **Secondary Dispatch**: Place 2 officers at secondary junctions for manual signaling adjustments.\n\n"
                "**Action Plan**: Signal offsets should be increased by +15 seconds along the main exit corridor."
            )
        elif 'diversion' in q or 'route' in q or 'road' in q:
            return (
                "### 🗺️ Traffic Diversion Advisory\n\n"
                "Due to the current congestion profile at **QueensStatueCircle**:\n"
                "1. **Primary Diversion**: Route outbound traffic via alternative segments with capacity utilization < 0.6.\n"
                "2. **Barricading**: Erect containment barriers 200m upstream to prevent gridlock. "
                f"The road closure probability is estimated at **{context.get('road_closure_probability', 0.0) * 100}%**.\n"
                "3. **Signal adjustment**: Trigger green wave synchronization along the diversion corridor."
            )
        else:
            return (
                "### 📊 Command Room Status Summary\n\n"
                f"Currently monitoring **{context.get('incident_count', 3)} active bottlenecks** with a global "
                f"congestion level of **{context.get('average_congestion', 4.2)}/10**.\n\n"
                "**Core Recommendations**:\n"
                "- Deploy officers proportionally to critical segments.\n"
                "- Calibrate Webster signal cycle times on corridors undergoing construction.\n"
                "Please query me for specific dispatch schedules or diversion maps."
            )

if __name__ == '__main__':
    import sys
    # Reconfigure stdout for unicode print support on windows terminal
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
        
    copilot = AstraCopilot()
    test_context = {
        'incident_count': 5,
        'average_congestion': 7.2,
        'bottlenecks': ['Tumkur Road', 'Bellary Road (Hebbal)'],
        'available_officers': 15,
        'predicted_duration_mins': 145.0,
        'road_closure_probability': 0.85,
        'priority': 'High',
        'manpower_recommendation': 'Deploy 4 officers to Hebbal Junction and 2 to Jalahalli Cross',
        'signal_strategy': 'Extend critical green phase by 30 seconds (Webster model)',
        'barricade_strategy': 'Erect containment barricades 200m upstream to initiate diversion route'
    }
    
    print("--- Testing Copilot Query (Manpower) ---")
    response_text = copilot.chat_with_copilot("What is the manpower dispatch recommendation for today's high priority incident?", test_context)
    print(response_text)

