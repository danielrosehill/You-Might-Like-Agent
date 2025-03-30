#!/usr/bin/env python3
"""
You Might Like - AI-powered recommendation assistant
Generates personalized content recommendations based on user preferences
"""

import os
import sys
import json
import time
import requests
import argparse
import datetime
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
from dotenv import load_dotenv
from pick import pick
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv()

# Initialize console for rich output
console = Console()

# Define content categories
CATEGORIES = [
    "Movies",
    "Netflix Shows",
    "Documentaries",
    "YouTube Channels",
    "Podcasts",
    "Books",
    "Online Courses",
    "People to Follow",
    "Blogs"
]

# Folder mapping for recommendations
FOLDER_MAPPING = {
    "Movies": "movies",
    "Netflix Shows": "netflix",
    "Documentaries": "documentaries",
    "YouTube Channels": "youtube",
    "Podcasts": "podcasts",
    "Books": "books",
    "Online Courses": "courses",
    "People to Follow": "people",
    "Blogs": "blogs"
}

# Current year for recency filtering
CURRENT_YEAR = datetime.datetime.now().year

class RecommendationAssistant:
    """AI-powered recommendation assistant that generates personalized content suggestions"""
    
    def __init__(self):
        """Initialize the recommendation assistant"""
        self.preferences = self._load_preferences()
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.tmdb_api_key = os.getenv("TMDB_API_KEY")
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        # Track recommendations to avoid duplicates
        self.recommended_titles = set()
        self.recommended_ids = set()
        
        # Validate API keys
        self._validate_api_keys()
        
        # Create recommendation directories if they don't exist
        self._create_directories()
    
    def _load_preferences(self) -> str:
        """Load user preferences from I-like.md file"""
        try:
            with open("I-like.md", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            console.print("[bold red]Error:[/bold red] I-like.md file not found.")
            console.print("Please create this file with your entertainment preferences.")
            sys.exit(1)
    
    def _validate_api_keys(self):
        """Validate that necessary API keys are present"""
        if not (self.openrouter_api_key or self.openai_api_key):
            console.print("[bold red]Error:[/bold red] No LLM API key found.")
            console.print("Please set either OPENROUTER_API_KEY or OPENAI_API_KEY in your .env file.")
            sys.exit(1)
            
        if not self.tmdb_api_key:
            console.print("[bold yellow]Warning:[/bold yellow] TMDB_API_KEY not found.")
            console.print("Movie and TV show validation will be limited.")
        
        search_api_available = False
        if self.tavily_api_key:
            search_api_available = True
        elif self.perplexity_api_key:
            search_api_available = True
            
        if not search_api_available:
            console.print("[bold yellow]Warning:[/bold yellow] No search API key found (Tavily or Perplexity).")
            console.print("Recommendations may not include the most recent content.")
    
    def _create_directories(self):
        """Create directories for storing recommendations"""
        for folder in FOLDER_MAPPING.values():
            Path(folder).mkdir(exist_ok=True)
    
    def select_categories(self) -> List[str]:
        """Let user select which categories to generate recommendations for"""
        title = "Select categories for recommendations (SPACE to select, ENTER to confirm):"
        selected = pick(CATEGORIES, title, multiselect=True, min_selection_count=1)
        return [category for category, _ in selected]
    
    def get_recommendation_count(self) -> int:
        """Get the number of recommendations to generate per category"""
        while True:
            try:
                count = input("\nHow many recommendations would you like per category? (default: 10): ")
                if not count:
                    return 10
                count = int(count)
                if count < 1:
                    console.print("[bold red]Error:[/bold red] Please enter a positive number.")
                    continue
                return count
            except ValueError:
                console.print("[bold red]Error:[/bold red] Please enter a valid number.")
    
    def call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API to generate recommendations"""
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "anthropic/claude-3-opus:beta",
            "messages": [
                {"role": "system", "content": "You are a helpful recommendation assistant that provides personalized content suggestions. Focus on recent content from the past 5 years, with a strong preference for content released in the last 1-2 years."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        
        if response.status_code != 200:
            console.print(f"[bold red]Error:[/bold red] API request failed with status code {response.status_code}")
            console.print(response.text)
            return ""
        
        return response.json()["choices"][0]["message"]["content"]
    
    def call_openai(self, prompt: str) -> str:
        """Call OpenAI API to generate recommendations"""
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful recommendation assistant that provides personalized content suggestions. Focus on recent content from the past 5 years, with a strong preference for content released in the last 1-2 years."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        
        if response.status_code != 200:
            console.print(f"[bold red]Error:[/bold red] API request failed with status code {response.status_code}")
            console.print(response.text)
            return ""
        
        return response.json()["choices"][0]["message"]["content"]
    
    def call_perplexity(self, query: str) -> str:
        """Call Perplexity Sonar API for up-to-date information"""
        if not self.perplexity_api_key:
            return ""
            
        headers = {
            "Authorization": f"Bearer {self.perplexity_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "query": query,
            "max_tokens": 1000
        }
        
        response = requests.post("https://api.perplexity.ai/sonar/v1/query", headers=headers, json=data)
        
        if response.status_code != 200:
            console.print(f"[bold yellow]Warning:[/bold yellow] Perplexity API request failed with status code {response.status_code}")
            return ""
        
        return response.json().get("text", "")
    
    def call_tavily(self, query: str) -> str:
        """Call Tavily API for up-to-date information"""
        if not self.tavily_api_key:
            return ""
            
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.tavily_api_key
        }
        
        data = {
            "query": query,
            "search_depth": "advanced",
            "include_domains": [],
            "exclude_domains": [],
            "max_results": 5
        }
        
        response = requests.post("https://api.tavily.com/search", headers=headers, json=data)
        
        if response.status_code != 200:
            console.print(f"[bold yellow]Warning:[/bold yellow] Tavily API request failed with status code {response.status_code}")
            return ""
        
        result = response.json()
        
        # Format the results
        formatted_result = ""
        for i, item in enumerate(result.get("results", [])):
            formatted_result += f"Source {i+1}: {item.get('title')}\n"
            formatted_result += f"URL: {item.get('url')}\n"
            formatted_result += f"Content: {item.get('content')}\n\n"
        
        return formatted_result
    
    def get_search_results(self, query: str) -> str:
        """Get search results from either Tavily or Perplexity"""
        if self.tavily_api_key:
            return self.call_tavily(query)
        elif self.perplexity_api_key:
            return self.call_perplexity(query)
        return ""
    
    def is_duplicate(self, title: str, tmdb_id: Optional[int] = None) -> bool:
        """Check if a recommendation is a duplicate based on title or TMDB ID"""
        # Normalize title for comparison
        normalized_title = title.lower().strip()
        
        # Check for exact title match
        if normalized_title in self.recommended_titles:
            return True
        
        # Check for TMDB ID match if available
        if tmdb_id and tmdb_id in self.recommended_ids:
            return True
        
        # Check for similar titles using fuzzy matching
        for existing_title in self.recommended_titles:
            # Simple similarity check - if titles are very similar
            if self._title_similarity(normalized_title, existing_title) > 0.8:
                return True
        
        return False
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles (0-1 scale)"""
        # Simple implementation - can be improved with more sophisticated algorithms
        if title1 == title2:
            return 1.0
            
        # Check if one is a substring of the other
        if title1 in title2 or title2 in title1:
            return 0.9
            
        # Calculate word overlap
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
            
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        return overlap / total
    
    def validate_tmdb(self, title: str, media_type: str) -> Dict[str, Any]:
        """Validate that a movie or TV show exists using TMDB API and check its recency"""
        if not self.tmdb_api_key:
            return {"valid": False, "data": {}}
            
        media_type_map = {
            "Movies": "movie",
            "Netflix Shows": "tv"
        }
        
        tmdb_type = media_type_map.get(media_type, "movie")
        
        params = {
            "api_key": self.tmdb_api_key,
            "query": title,
            "language": "en-US",
            "page": 1,
            "include_adult": "false"
        }
        
        response = requests.get(f"https://api.themoviedb.org/3/search/{tmdb_type}", params=params)
        
        if response.status_code != 200:
            return {"valid": False, "data": {}}
        
        results = response.json().get("results", [])
        
        if not results:
            return {"valid": False, "data": {}}
        
        # Filter for recent results (within the last 5 years)
        recent_results = []
        for result in results:
            date_field = "release_date" if tmdb_type == "movie" else "first_air_date"
            if date_field in result and result[date_field]:
                try:
                    year = int(result[date_field].split("-")[0])
                    if CURRENT_YEAR - year <= 5:
                        recent_results.append(result)
                except (ValueError, IndexError):
                    pass
        
        # If no recent results, use the first result
        if not recent_results and results:
            recent_results = [results[0]]
        
        # Sort by date (newest first)
        if recent_results:
            date_field = "release_date" if tmdb_type == "movie" else "first_air_date"
            recent_results.sort(key=lambda x: x.get(date_field, ""), reverse=True)
        
        if not recent_results:
            return {"valid": False, "data": {}}
        
        # Get the first result
        first_result = recent_results[0]
        
        # Check if this is a duplicate by TMDB ID
        if self.is_duplicate(first_result.get("title", first_result.get("name", "")), first_result.get("id")):
            return {"valid": False, "data": {}, "reason": "duplicate"}
        
        # Get trailer if available
        trailer_url = ""
        if self.tmdb_api_key:
            videos_response = requests.get(
                f"https://api.themoviedb.org/3/{tmdb_type}/{first_result['id']}/videos",
                params={"api_key": self.tmdb_api_key}
            )
            
            if videos_response.status_code == 200:
                videos = videos_response.json().get("results", [])
                trailers = [v for v in videos if v.get("type") == "Trailer" and v.get("site") == "YouTube"]
                
                if trailers:
                    trailer_url = f"https://www.youtube.com/watch?v={trailers[0]['key']}"
        
        # Check if Netflix show is currently available (for Netflix category)
        netflix_available = False
        if media_type == "Netflix Shows" and self.tmdb_api_key:
            providers_response = requests.get(
                f"https://api.themoviedb.org/3/{tmdb_type}/{first_result['id']}/watch/providers",
                params={"api_key": self.tmdb_api_key}
            )
            
            if providers_response.status_code == 200:
                providers = providers_response.json().get("results", {})
                # Check Israel providers first, then US
                for country in ["IL", "US"]:
                    if country in providers:
                        flatrate = providers[country].get("flatrate", [])
                        for provider in flatrate:
                            if provider.get("provider_name") == "Netflix":
                                netflix_available = True
                                break
        
        # For Netflix shows, only return valid if available on Netflix
        if media_type == "Netflix Shows" and not netflix_available:
            return {"valid": False, "data": {}}
        
        # Add to recommended titles and IDs
        title_to_add = first_result.get("title", first_result.get("name", "")).lower().strip()
        self.recommended_titles.add(title_to_add)
        if first_result.get("id"):
            self.recommended_ids.add(first_result.get("id"))
        
        return {
            "valid": True,
            "data": {
                "id": first_result.get("id"),
                "title": first_result.get("title", first_result.get("name", "")),
                "overview": first_result.get("overview", ""),
                "release_date": first_result.get("release_date", first_result.get("first_air_date", "")),
                "poster_path": first_result.get("poster_path", ""),
                "trailer_url": trailer_url,
                "year": first_result.get("release_date", first_result.get("first_air_date", "")).split("-")[0] if first_result.get("release_date", first_result.get("first_air_date", "")) else ""
            }
        }
    
    def generate_prompt(self, category: str, count: int) -> str:
        """Generate a prompt for the AI model based on user preferences"""
        current_year = datetime.datetime.now().year
        
        # Add exclusion list for already recommended titles
        exclusion_list = ""
        if self.recommended_titles:
            exclusion_list = "IMPORTANT: Do NOT recommend any of these titles as they've already been recommended in other categories:\n"
            exclusion_list += ", ".join(f'"{title}"' for title in self.recommended_titles)
            exclusion_list += "\n\n"
        
        return f"""Based on the following user preferences, generate {count} personalized {category} recommendations:

{self.preferences}

{exclusion_list}IMPORTANT REQUIREMENTS:
1. Focus ONLY on content released in the last 5 years ({current_year-5} to {current_year}), with a strong preference for content from the last 1-2 years.
2. For Netflix shows, only recommend shows currently available on Netflix.
3. For all recommendations, prioritize recency and relevance to the user's interests.
4. The user is based in Israel, so consider content availability in that region.
5. For movies and shows, focus on true stories, geopolitics, intelligence, journalism, and Israeli/Jewish themes.
6. Avoid science fiction, gory war movies, and World War II content.
7. Each recommendation MUST be unique across all categories - do not recommend the same content in multiple categories.

For each recommendation, provide:
1. Title
2. Creator/Author/Director
3. Year (MUST be between {current_year-5} and {current_year})
4. Brief description of the content
5. Why it matches the user's preferences
6. Where to find it (platform, website, etc.)
7. A link to more information or content (trailer, website, etc.)

Format each recommendation as a JSON object with these fields:
- title: The title of the recommendation
- creator: The creator, author, or director
- year: The release year (MUST be between {current_year-5} and {current_year})
- description: A brief description of the content
- reason: Why this matches the user's preferences
- where_to_find: Where to access this content
- link: A link to more information (trailer, website, etc.)

Return ONLY a valid JSON array with {count} recommendations, nothing else.
"""
    
    def parse_recommendations(self, response: str) -> List[Dict[str, str]]:
        """Parse the AI response into a list of recommendation objects"""
        # Extract JSON from the response
        try:
            # Find JSON array in the response
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            
            if json_start == -1 or json_end == 0:
                console.print("[bold red]Error:[/bold red] Could not find JSON array in response.")
                return []
            
            json_str = response[json_start:json_end]
            recommendations = json.loads(json_str)
            
            # Filter out recommendations older than 5 years
            current_year = datetime.datetime.now().year
            filtered_recommendations = []
            
            for rec in recommendations:
                try:
                    # Check if it's a duplicate
                    if "title" in rec and self.is_duplicate(rec["title"]):
                        console.print(f"[bold yellow]Filtering out:[/bold yellow] {rec.get('title', 'Unknown')} - duplicate recommendation")
                        continue
                        
                    if "year" in rec and rec["year"]:
                        year = int(rec["year"])
                        if current_year - year <= 5:
                            # Add to recommended titles
                            if "title" in rec:
                                self.recommended_titles.add(rec["title"].lower().strip())
                            filtered_recommendations.append(rec)
                        else:
                            console.print(f"[bold yellow]Filtering out:[/bold yellow] {rec.get('title', 'Unknown')} ({rec.get('year', 'Unknown')}) - too old")
                    else:
                        # Add to recommended titles
                        if "title" in rec:
                            self.recommended_titles.add(rec["title"].lower().strip())
                        filtered_recommendations.append(rec)
                except (ValueError, TypeError):
                    # Add to recommended titles
                    if "title" in rec:
                        self.recommended_titles.add(rec["title"].lower().strip())
                    filtered_recommendations.append(rec)
            
            return filtered_recommendations
            
        except json.JSONDecodeError:
            console.print("[bold red]Error:[/bold red] Failed to parse recommendations as JSON.")
            console.print("Response:", response)
            return []
    
    def enrich_recommendation(self, recommendation: Dict[str, str], category: str) -> Dict[str, str]:
        """Enrich recommendation with additional information if available"""
        enriched = recommendation.copy()
        
        # Create search query
        search_query = f"{recommendation['title']} {recommendation.get('creator', '')} {recommendation.get('year', '')} {category.lower()} recent information"
        
        # Validate movies and TV shows with TMDB
        if category in ["Movies", "Netflix Shows"]:
            validation = self.validate_tmdb(recommendation["title"], category)
            
            if validation["valid"]:
                data = validation["data"]
                
                # Add or update fields
                if "trailer_url" in data and data["trailer_url"]:
                    enriched["link"] = data["trailer_url"]
                
                if "overview" in data and data["overview"]:
                    # Only use TMDB overview if it's more detailed
                    if len(data["overview"]) > len(enriched.get("description", "")):
                        enriched["description"] = data["overview"]
                
                if "year" in data and data["year"]:
                    enriched["year"] = data["year"]
                
                # Add poster path if available
                if "poster_path" in data and data["poster_path"]:
                    enriched["poster"] = f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
            elif validation.get("reason") == "duplicate":
                # If this is a duplicate, try to get a different recommendation
                console.print(f"[bold yellow]Skipping:[/bold yellow] {recommendation.get('title', 'Unknown')} - duplicate recommendation")
                return {"skip": True}
            else:
                # If TMDB validation fails, try to get more recent information
                search_results = self.get_search_results(search_query)
                if search_results:
                    enriched["search_results"] = search_results
        else:
            # For non-movie/TV content, get up-to-date information
            search_results = self.get_search_results(search_query)
            if search_results:
                enriched["search_results"] = search_results
            
            # Add to recommended titles for non-TMDB content
            if "title" in recommendation:
                self.recommended_titles.add(recommendation["title"].lower().strip())
        
        return enriched
    
    def create_markdown(self, recommendation: Dict[str, str], category: str) -> str:
        """Create a markdown file for a recommendation"""
        title = recommendation.get("title", "Untitled")
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()
        
        folder = FOLDER_MAPPING.get(category, "other")
        filename = f"{folder}/{safe_title}.md"
        
        # Create markdown content
        content = f"# {title}\n\n"
        
        if "creator" in recommendation and recommendation["creator"]:
            content += f"**By:** {recommendation['creator']}\n\n"
        
        if "year" in recommendation and recommendation["year"]:
            content += f"**Year:** {recommendation['year']}\n\n"
        
        if "description" in recommendation and recommendation["description"]:
            content += f"## Description\n\n{recommendation['description']}\n\n"
        
        if "reason" in recommendation and recommendation["reason"]:
            content += f"## Why You Might Like It\n\n{recommendation['reason']}\n\n"
        
        if "where_to_find" in recommendation and recommendation["where_to_find"]:
            content += f"## Where to Find It\n\n{recommendation['where_to_find']}\n\n"
        
        if "link" in recommendation and recommendation["link"]:
            if category == "Movies" and "youtube.com" in recommendation["link"].lower():
                content += f"## Trailer\n\n[Watch Trailer]({recommendation['link']})\n\n"
            else:
                content += f"## Link\n\n[More Information]({recommendation['link']})\n\n"
        
        if "poster" in recommendation and recommendation["poster"]:
            content += f"![Poster]({recommendation['poster']})\n\n"
        
        if "search_results" in recommendation and recommendation["search_results"]:
            content += f"## Additional Information\n\n{recommendation['search_results']}\n\n"
        
        content += f"*Generated on {datetime.datetime.now().strftime('%Y-%m-%d')}*"
        
        # Write to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        
        return filename
    
    def generate_recommendations(self, category: str, count: int) -> None:
        """Generate recommendations for a specific category"""
        console.print(f"\n[bold blue]Generating {count} {category} recommendations...[/bold blue]")
        
        # Generate prompt
        prompt = self.generate_prompt(category, count)
        
        # Call AI API
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Thinking about {category} you might like...", total=None)
            
            if self.openrouter_api_key:
                response = self.call_openrouter(prompt)
            else:
                response = self.call_openai(prompt)
                
            progress.update(task, completed=True)
        
        if not response:
            console.print(f"[bold red]Error:[/bold red] Failed to generate {category} recommendations.")
            return
        
        # Parse recommendations
        recommendations = self.parse_recommendations(response)
        
        if not recommendations:
            console.print(f"[bold red]Error:[/bold red] No valid {category} recommendations found in response.")
            return
        
        # Process each recommendation
        successful_recommendations = 0
        for i, recommendation in enumerate(recommendations):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Processing recommendation {i+1}/{len(recommendations)}: {recommendation.get('title', 'Untitled')}", total=None)
                
                # Enrich recommendation
                enriched = self.enrich_recommendation(recommendation, category)
                
                # Skip if duplicate
                if enriched.get("skip", False):
                    progress.update(task, completed=True)
                    continue
                
                # Create markdown file
                filename = self.create_markdown(enriched, category)
                successful_recommendations += 1
                
                progress.update(task, completed=True)
            
            console.print(f"  ✓ Created [cyan]{filename}[/cyan]")
            
            # Small delay to avoid rate limiting
            if i < len(recommendations) - 1:
                time.sleep(0.5)
        
        # If we didn't get enough recommendations, log a warning
        if successful_recommendations < count:
            console.print(f"[bold yellow]Warning:[/bold yellow] Only generated {successful_recommendations}/{count} recommendations for {category} due to duplicates or filtering.")
    
    def run(self):
        """Run the recommendation assistant"""
        console.print("[bold green]===== You Might Like =====[/bold green]")
        console.print("AI-powered recommendation assistant\n")
        
        # Select categories
        selected_categories = self.select_categories()
        
        # Get recommendation count
        count = self.get_recommendation_count()
        
        console.print(f"\n[bold]Generating {count} recommendations for each of these categories:[/bold]")
        for category in selected_categories:
            console.print(f"  • {category}")
        
        # Generate recommendations for each category
        for category in selected_categories:
            self.generate_recommendations(category, count)
        
        console.print("\n[bold green]Done![/bold green] Your personalized recommendations are ready.")
        console.print("Check the category folders for markdown files with your recommendations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-powered recommendation assistant")
    parser.add_argument("--version", action="version", version="You Might Like v1.0.0")
    args = parser.parse_args()
    
    try:
        assistant = RecommendationAssistant()
        assistant.run()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Process interrupted by user.[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)
