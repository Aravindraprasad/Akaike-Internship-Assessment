import requests
from bs4 import BeautifulSoup
import nltk
from newspaper import Article
from googlesearch import search
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import json
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import tempfile
from pydub import AudioSegment

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class NewsScraper:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize summarization model
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def get_company_news_sources(self, company_name):
        """Get direct news sources for a company"""
        company_lower = company_name.lower()
        
        # Common financial news sources
        common_sources = [
            "https://www.reuters.com/business/",
            "https://www.bloomberg.com/",
            "https://www.cnbc.com/",
            "https://www.ft.com/",
            "https://www.wsj.com/",
            "https://finance.yahoo.com/"
        ]
        
        # Company-specific sources
        company_sources = {
            "apple": [
                "https://www.apple.com/newsroom/",
                "https://9to5mac.com/",
                "https://www.macrumors.com/"
            ],
            "microsoft": [
                "https://news.microsoft.com/",
                "https://techcrunch.com/tag/microsoft/",
                "https://www.theverge.com/microsoft"
            ],
            "google": [
                "https://blog.google/",
                "https://techcrunch.com/tag/google/",
                "https://www.theverge.com/google"
            ],
            "amazon": [
                "https://press.aboutamazon.com/",
                "https://techcrunch.com/tag/amazon/",
                "https://www.aboutamazon.com/news"
            ],
            "tesla": [
                "https://www.tesla.com/blog",
                "https://electrek.co/guides/tesla/",
                "https://insideevs.com/tag/tesla/"
            ]
            # Add more companies as needed
        }
        
        # Get sources for this company or return common sources
        specific_sources = company_sources.get(company_lower, [])
        return specific_sources + common_sources
        
    def search_news_articles(self, company_name, num_articles=15):
        """Search for news articles about the company"""
        query = f"{company_name} company news"
        
        try:
            # First try Google search with increased timeout
            search_results = search(query, num_results=num_articles*2, timeout=10)
            
            # Filter non-JavaScript websites
            valid_urls = []
            for url in search_results:
                if not self._is_js_heavy_site(url) and len(valid_urls) < num_articles:
                    valid_urls.append(url)
            
            # If we found enough URLs, return them
            if len(valid_urls) >= 3:
                return valid_urls[:num_articles]
                
            # Otherwise, fall back to direct news sources
            company_sources = self.get_company_news_sources(company_name)
            return company_sources[:num_articles]
            
        except Exception as e:
            print(f"Error searching for news: {str(e)}")
            # Return direct news sources as fallback
            return self.get_company_news_sources(company_name)[:num_articles]
    
    def _is_js_heavy_site(self, url):
        """Check if the site is JS-heavy and might be difficult to scrape with BeautifulSoup"""
        js_heavy_domains = ['twitter.com', 'facebook.com', 'instagram.com', 'linkedin.com']
        return any(domain in url for domain in js_heavy_domains)
    
    def extract_article_content(self, url):
        """Extract article content using newspaper3k"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()  # This does the keyword extraction and summarization
            
            return {
                "url": url,
                "title": article.title,
                "text": article.text,
                "summary": article.summary,
                "keywords": article.keywords,
                "publish_date": article.publish_date.strftime('%Y-%m-%d') if article.publish_date else None,
                "authors": article.authors
            }
        except Exception as e:
            print(f"Error extracting content from {url}: {str(e)}")
            return None
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of the text"""
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        
        if compound_score >= 0.05:
            sentiment = "Positive"
        elif compound_score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return {
            "sentiment": sentiment,
            "scores": sentiment_scores
        }
    
    def extract_topics(self, text, num_topics=5):
        """Extract main topics from the text"""
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token.isalpha() and token not in self.stop_words]
        
        # Get most common words as topics
        topic_counter = Counter(tokens)
        topics = [topic for topic, _ in topic_counter.most_common(num_topics)]
        
        return topics
    
    def get_improved_summary(self, text, max_length=150):
        """Generate improved summary using the BART model"""
        if len(text.split()) < 50:  # If text is already short
            return text
            
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
            return summary
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            # Fallback to a simple extractive summary
            sentences = nltk.sent_tokenize(text)
            return ' '.join(sentences[:3])  # First 3 sentences
    
    def perform_comparative_analysis(self, articles):
        """Compare sentiment and topics across multiple articles"""
        sentiment_distribution = {"Positive": 0, "Negative": 0, "Neutral": 0}
        all_topics = []
        
        for article in articles:
            sentiment_distribution[article["Sentiment"]] += 1
            all_topics.extend(article["Topics"])
        
        # Find common topics
        topic_counter = Counter(all_topics)
        common_topics = [topic for topic, count in topic_counter.items() if count > 1]
        
        # Generate comparison insights
        coverage_differences = []
        for i in range(len(articles)):
            for j in range(i+1, len(articles)):
                comparison = {
                    "Comparison": f"Article {i+1} focuses on {', '.join(articles[i]['Topics'][:2])}, while Article {j+1} focuses on {', '.join(articles[j]['Topics'][:2])}.",
                    "Impact": self._generate_impact_statement(articles[i], articles[j])
                }
                coverage_differences.append(comparison)
        
        # Analyze topic overlap
        all_unique_topics = list(set(all_topics))
        article_topics = [set(article["Topics"]) for article in articles]
        
        topic_overlap = {
            "Common Topics": common_topics,
            "All Topics": all_unique_topics
        }
        
        # For each article, identify unique topics
        for i, topics in enumerate(article_topics):
            unique_topics = topics - set().union(*[article_topics[j] for j in range(len(article_topics)) if j != i])
            topic_overlap[f"Unique Topics in Article {i+1}"] = list(unique_topics)
        
        return {
            "Sentiment Distribution": sentiment_distribution,
            "Coverage Differences": coverage_differences[:5],  # Limit to top 5 differences
            "Topic Overlap": topic_overlap
        }
    
    def _generate_impact_statement(self, article1, article2):
        """Generate an impact statement comparing two articles"""
        if article1["Sentiment"] == article2["Sentiment"]:
            return f"Both articles share a {article1['Sentiment'].lower()} sentiment, reinforcing the overall market perception."
        else:
            return f"The {article1['Sentiment'].lower()} tone in the first article contrasts with the {article2['Sentiment'].lower()} tone in the second, suggesting mixed market signals."
    
    def generate_final_sentiment(self, comparative_analysis):
        """Generate a final sentiment analysis statement"""
        sentiment_dist = comparative_analysis["Sentiment Distribution"]
        total = sum(sentiment_dist.values())
        
        if sentiment_dist["Positive"] > sentiment_dist["Negative"] + sentiment_dist["Neutral"]:
            trend = "positive"
            expectation = "Potential stock growth expected"
        elif sentiment_dist["Negative"] > sentiment_dist["Positive"] + sentiment_dist["Neutral"]:
            trend = "negative"
            expectation = "Potential market challenges ahead"
        elif sentiment_dist["Positive"] > sentiment_dist["Negative"]:
            trend = "slightly positive"
            expectation = "Cautious optimism in market outlook"
        elif sentiment_dist["Negative"] > sentiment_dist["Positive"]:
            trend = "slightly negative"
            expectation = "Some concerns in market perception"
        else:
            trend = "mixed"
            expectation = "Unclear market direction at this time"
        
        return f"The company's latest news coverage is mostly {trend}. {expectation}."


class TextToSpeech:
    def __init__(self):
        # Use a more widely available TTS solution
        try:
            from transformers import pipeline
            self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
            # Use gTTS instead of indicTTS
            from gtts import gTTS
            self.tts_available = True
        except Exception as e:
            print(f"Error initializing TTS: {str(e)}")
            self.tts_available = False
    
    def translate_to_hindi(self, text):
        """Translate English text to Hindi"""
        try:
            if hasattr(self, 'translator'):
                hindi_text = self.translator(text, max_length=512)[0]['translation_text']
                return hindi_text
        except Exception as e:
            print(f"Error in translation: {str(e)}")
        # Return a simple Hindi fallback message
        return "यह एक समाचार सारांश है। अनुवाद उपलब्ध नहीं है।"
    
    def generate_speech(self, text):
        """Generate Hindi speech from text"""
        if not self.tts_available:
            print("TTS not available")
            return None
        
        try:
            # Use gTTS instead
            from gtts import gTTS
            import tempfile
            
            # Generate temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            output_path = temp_file.name
            
            # Convert text to speech
            tts = gTTS(text=text, lang='hi', slow=False)
            tts.save(output_path)
            
            return output_path
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return None


def process_company_news(company_name):
    """Process news for the given company"""
    scraper = NewsScraper()
    tts = TextToSpeech()
    
    # Try to get real news articles
    urls = scraper.search_news_articles(company_name, num_articles=10)
    articles_data = []
    
    # Track if we're using mock data
    using_mock_data = False
    
    # Try to process real articles
    for url in urls:
        try:
            article = scraper.extract_article_content(url)
            if article:
                # Process article
                sentiment_analysis = scraper.analyze_sentiment(article["text"])
                topics = scraper.extract_topics(article["text"])
                improved_summary = scraper.get_improved_summary(article["text"])
                
                # Create article_info dictionary - this was missing/incomplete in original code
                article_info = {
                    "Title": article["title"],
                    "Summary": improved_summary,
                    "Sentiment": sentiment_analysis["sentiment"],
                    "Topics": topics if topics else article["keywords"]
                }
                articles_data.append(article_info)
        except Exception as e:
            print(f"Error processing article from {url}: {str(e)}")
            continue
    
    # If no articles were successfully processed, use mock data
    if not articles_data:
        using_mock_data = True
        mock_articles = get_mock_news_data(company_name)
        for article in mock_articles:
            # Analyze sentiment
            sentiment_analysis = scraper.analyze_sentiment(article["text"])
            
            # Extract topics
            topics = scraper.extract_topics(article["text"])
            
            article_info = {
                "Title": article["title"],
                "Summary": article["summary"],
                "Sentiment": sentiment_analysis["sentiment"],
                "Topics": topics if topics else article["keywords"]
            }
            articles_data.append(article_info)
    
    # Perform comparative analysis
    if len(articles_data) >= 2:
        comparative_analysis = scraper.perform_comparative_analysis(articles_data)
        final_sentiment = scraper.generate_final_sentiment(comparative_analysis)
        if using_mock_data:
            final_sentiment += " (Based on simulated data due to connection issues)"
    else:
        comparative_analysis = {
            "Sentiment Distribution": {"Positive": 1, "Neutral": 1, "Negative": 0},
            "Coverage Differences": [],
            "Topic Overlap": {"Common Topics": [], "All Topics": [t for article in articles_data for t in article["Topics"]]}
        }
        final_sentiment = "Limited article data available for comprehensive analysis."
    
    # Generate summary for TTS
    summary_for_tts = f"{company_name} company news analysis. {final_sentiment}"
    
    # Convert to Hindi
    hindi_summary = tts.translate_to_hindi(summary_for_tts)
    
    # Generate speech
    audio_path = tts.generate_speech(hindi_summary)
    
    # Prepare final output
    result = {
        "Company": company_name,
        "Articles": articles_data,
        "Comparative Sentiment Score": comparative_analysis,
        "Final Sentiment Analysis": final_sentiment,
        "Hindi Summary": hindi_summary,
        "Audio": audio_path
    }
    
    return result

def get_mock_news_data(company_name):
    """Provide mock news data when real scraping fails"""
    mock_articles = [
        {
            "url": "https://example.com/news1",
            "title": f"{company_name} Announces Quarterly Results",
            "text": f"{company_name} today announced its quarterly financial results, exceeding analyst expectations. The company reported strong growth in its core business segments, with revenue increasing by 12% year-over-year. The CEO highlighted new product innovations and market expansion strategies as key drivers of the company's success.",
            "summary": f"{company_name} announced quarterly results exceeding expectations with 12% YoY revenue growth.",
            "keywords": ["financial", "quarterly results", "growth", "revenue"],
            "publish_date": "2025-03-15",
            "authors": ["Financial Analyst"]
        },
        {
            "url": "https://example.com/news2",
            "title": f"{company_name} Expands Into New Markets",
            "text": f"{company_name} is expanding its operations into emerging markets, according to industry sources. The strategic move aims to capture growing demand in regions with increasing technological adoption. Analysts view this expansion positively, suggesting it could increase the company's global market share by up to 5% within the next fiscal year.",
            "summary": f"{company_name} is expanding into emerging markets, potentially increasing global market share by 5%.",
            "keywords": ["expansion", "emerging markets", "strategy", "growth"],
            "publish_date": "2025-03-10",
            "authors": ["Business Reporter"]
        },
        {
            "url": "https://example.com/news3",
            "title": f"New Product Launch From {company_name}",
            "text": f"{company_name} has unveiled its latest product line, featuring cutting-edge technology and improved user experience. The announcement came during a special event that highlighted the company's commitment to innovation. Industry experts note that these new offerings could strengthen the company's competitive position against rivals in the sector.",
            "summary": f"{company_name} launched a new product line featuring advanced technology and improved user experience.",
            "keywords": ["product launch", "innovation", "technology", "competition"],
            "publish_date": "2025-03-05",
            "authors": ["Tech Reporter"]
        }
    ]
    
    return mock_articles