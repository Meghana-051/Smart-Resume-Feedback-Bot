import streamlit as st
import spacy
import pdfplumber
import docx
import re
from collections import Counter
import datetime
import tempfile
import base64
from typing import Dict, List, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="Resume Analyzer Pro", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load spaCy model
@st.cache_resource
def load_nlp_model() -> Optional[spacy.language.Language]:
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy English model not found. Please install it with: python -m spacy download en_core_web_sm")
        return None

nlp = load_nlp_model()

class ResumeParser:
    """Class to handle resume parsing and analysis"""
    
    @staticmethod
    def extract_text(file) -> str:
        """Extract text from PDF or DOCX files"""
        if file.type == "application/pdf":
            return ResumeParser._extract_text_from_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return ResumeParser._extract_text_from_docx(file)
        else:
            raise ValueError("Unsupported file format")

    @staticmethod
    def _extract_text_from_pdf(file) -> str:
        """Extract text from PDF file"""
        try:
            with pdfplumber.open(file) as pdf:
                text = ''
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""

    @staticmethod
    def _extract_text_from_docx(file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file)
            text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    text.append(para.text.strip())
            return '\n'.join(text)
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""

class ResumeAnalyzer:
    """Class to analyze resume content"""
    
    @staticmethod
    def extract_contact_info(text: str) -> Dict[str, List[str]]:
        """Extract contact information from resume text"""
        contact_info = {}
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        contact_info['emails'] = re.findall(email_pattern, text)
        
        # Phone pattern
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        contact_info['phones'] = [phone[0] + phone[1] if isinstance(phone, tuple) else phone for phone in phones]
        
        # LinkedIn pattern
        linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+'
        contact_info['linkedin'] = re.findall(linkedin_pattern, text, re.IGNORECASE)
        
        # GitHub pattern
        github_pattern = r'(?:https?://)?(?:www\.)?github\.com/[\w-]+'
        contact_info['github'] = re.findall(github_pattern, text, re.IGNORECASE)
        
        return contact_info

    @staticmethod
    def analyze_structure(text: str) -> Dict[str, bool]:
        """Analyze resume structure and sections"""
        sections = {
            'contact': ['contact', 'personal information', 'details'],
            'summary': ['summary', 'objective', 'profile', 'about'],
            'experience': ['experience', 'work experience', 'employment', 'career history'],
            'education': ['education', 'academic', 'qualifications', 'degree'],
            'skills': ['skills', 'technical skills', 'competencies', 'expertise'],
            'projects': ['projects', 'portfolio', 'work samples'],
            'certifications': ['certifications', 'certificates', 'licenses'],
            'achievements': ['achievements', 'accomplishments', 'awards', 'honors'],
            'languages': ['languages', 'linguistic', 'multilingual'],
            'interests': ['interests', 'hobbies', 'activities']
        }
        
        found_sections = {}
        text_lower = text.lower()
        
        for section, keywords in sections.items():
            found_sections[section] = any(keyword in text_lower for keyword in keywords)
        
        return found_sections

    @staticmethod
    def extract_skills(text: str) -> Dict[str, List[str]]:
        """Enhanced skill extraction with categorization"""
        skills_database = {
            'Programming Languages': [
                'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift',
                'kotlin', 'typescript', 'scala', 'r', 'matlab', 'perl', 'shell', 'bash'
            ],
            'Web Technologies': [
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask',
                'spring', 'asp.net', 'bootstrap', 'jquery', 'webpack', 'sass', 'less'
            ],
            'Databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle', 'redis', 'cassandra',
                'dynamodb', 'elasticsearch', 'neo4j'
            ],
            'Cloud & DevOps': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible',
                'linux', 'unix', 'ci/cd', 'git', 'github', 'gitlab', 'bitbucket'
            ],
            'Data Science & AI': [
                'machine learning', 'deep learning', 'data science', 'data analysis', 'tensorflow',
                'pytorch', 'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'plotly',
                'tableau', 'power bi', 'spark', 'hadoop', 'kafka'
            ],
            'Mobile Development': [
                'android', 'ios', 'react native', 'flutter', 'xamarin', 'ionic'
            ],
            'Soft Skills': [
                'communication', 'teamwork', 'leadership', 'problem solving', 'critical thinking',
                'time management', 'adaptability', 'creativity', 'collaboration', 'attention to detail'
            ],
            'Project Management': [
                'agile', 'scrum', 'kanban', 'jira', 'trello', 'asana', 'monday', 'project management'
            ]
        }
        
        found_skills = {}
        text_lower = text.lower()
        
        for category, skills_list in skills_database.items():
            category_skills = [skill for skill in skills_list if skill.lower() in text_lower]
            if category_skills:
                found_skills[category] = category_skills
        
        return found_skills

    @staticmethod
    def calculate_score(text: str, contact_info: Dict, sections: Dict, skills: Dict) -> Tuple[int, Dict]:
        """Calculate resume score with detailed breakdown"""
        score = 0
        scoring_details = {}
        
        # Word count scoring (0-15 points)
        word_count = len(text.split())
        if word_count < 200:
            word_score = 5
        elif word_count < 400:
            word_score = 10
        elif word_count <= 800:
            word_score = 15
        else:
            word_score = 10  # Too long
        score += word_score
        scoring_details['Word Count'] = f"{word_score}/15"
        
        # Contact information scoring (0-10 points)
        contact_score = 0
        if contact_info['emails']:
            contact_score += 4
        if contact_info['phones']:
            contact_score += 3
        if contact_info['linkedin']:
            contact_score += 2
        if contact_info['github']:
            contact_score += 1
        score += contact_score
        scoring_details['Contact Info'] = f"{contact_score}/10"
        
        # Sections scoring (0-25 points)
        essential_sections = ['experience', 'education', 'skills']
        important_sections = ['summary', 'projects', 'certifications']
        
        section_score = sum(6 for section in essential_sections if sections.get(section, False))
        section_score += sum(2 for section in important_sections if sections.get(section, False))
        
        section_score = min(section_score, 25)
        score += section_score
        scoring_details['Sections'] = f"{section_score}/25"
        
        # Skills scoring (0-30 points)
        total_skills = sum(len(skills_list) for skills_list in skills.values())
        skills_score = min(total_skills * 2, 30)
        score += skills_score
        scoring_details['Skills'] = f"{skills_score}/30"
        
        # Format and readability (0-20 points)
        format_score = 0
        
        # Check for bullet points
        if '•' in text or '-' in text or '*' in text:
            format_score += 5
        
        # Check for proper capitalization
        sentences = re.split(r'[.!?]+', text)
        proper_caps = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
        if proper_caps / max(len(sentences), 1) > 0.7:
            format_score += 5
        
        # Check for excessive repetition
        words = text.lower().split()
        word_counts = Counter(words)
        most_common = word_counts.most_common(10)
        if not any(count > len(words) * 0.1 for word, count in most_common):
            format_score += 5
        
        # Check for section headers (all caps or title case)
        if re.search(r'\n[A-Z\s]{3,}\n', text) or re.search(r'\n[A-Z][a-z\s]+\n', text):
            format_score += 5
        
        score += format_score
        scoring_details['Format'] = f"{format_score}/20"
        
        return min(score, 100), scoring_details

    @staticmethod
    def generate_suggestions(text: str, contact_info: Dict, sections: Dict, skills: Dict, score: int) -> List[Dict]:
        """Generate comprehensive suggestions based on analysis"""
        suggestions = []
        word_count = len(text.split())
        
        # Word count suggestions
        if word_count < 200:
            suggestions.append({
                'type': 'critical',
                'title': 'Resume Too Short',
                'message': f'Your resume has only {word_count} words. Aim for 300-600 words.'
            })
        elif word_count > 800:
            suggestions.append({
                'type': 'warning',
                'title': 'Resume Too Long',
                'message': f'Your resume has {word_count} words. Consider condensing it to 300-600 words.'
            })
        
        # Contact information suggestions
        if not contact_info['emails']:
            suggestions.append({
                'type': 'critical',
                'title': 'Missing Email',
                'message': 'Include a professional email address.'
            })
        
        if not contact_info['phones']:
            suggestions.append({
                'type': 'warning',
                'title': 'Missing Phone Number',
                'message': 'Add a phone number for easier contact.'
            })
        
        if not contact_info['linkedin']:
            suggestions.append({
                'type': 'info',
                'title': 'Add LinkedIn Profile',
                'message': 'Include your LinkedIn profile URL.'
            })
        
        # Section suggestions
        if not sections.get('summary', False):
            suggestions.append({
                'type': 'warning',
                'title': 'Missing Professional Summary',
                'message': 'Add a brief professional summary at the beginning.'
            })
        
        if not sections.get('experience', False):
            suggestions.append({
                'type': 'critical',
                'title': 'Missing Work Experience',
                'message': 'Include your work experience with details.'
            })
        
        if not sections.get('education', False):
            suggestions.append({
                'type': 'critical',
                'title': 'Missing Education',
                'message': 'Add your educational background.'
            })
        
        if not sections.get('projects', False):
            suggestions.append({
                'type': 'info',
                'title': 'Consider Adding Projects',
                'message': 'Include relevant projects to showcase skills.'
            })
        
        # Skills suggestions
        total_skills = sum(len(skills_list) for skills_list in skills.values())
        if total_skills < 5:
            suggestions.append({
                'type': 'warning',
                'title': 'Limited Skills Listed',
                'message': f'You have only {total_skills} skills identified.'
            })
        
        # Technical skills suggestions
        if 'Programming Languages' not in skills and 'Web Technologies' not in skills:
            suggestions.append({
                'type': 'info',
                'title': 'Technical Skills',
                'message': 'If applicable, add programming languages or technical skills.'
            })
        
        # Format suggestions
        if '•' not in text and '-' not in text and '*' not in text:
            suggestions.append({
                'type': 'info',
                'title': 'Use Bullet Points',
                'message': 'Use bullet points for better readability.'
            })
        
        # Score-based suggestions
        if score < 40:
            suggestions.append({
                'type': 'critical',
                'title': 'Significant Improvements Needed',
                'message': 'Your resume needs major improvements.'
            })
        elif score < 70:
            suggestions.append({
                'type': 'warning',
                'title': 'Good Foundation, Needs Enhancement',
                'message': 'Your resume could benefit from additional details.'
            })
        else:
            suggestions.append({
                'type': 'success',
                'title': 'Well-Structured Resume',
                'message': 'Your resume is well-structured!'
            })
        
        return suggestions

    @staticmethod
    def extract_experience(text: str) -> List[Dict]:
        """Extract work experience information"""
        # This is a simplified version - could be enhanced with NLP
        experience = []
        lines = text.split('\n')
        
        current_job = None
        for line in lines:
            # Simple pattern matching for job titles and companies
            if re.search(r'\b(?:Senior|Junior|Lead|Manager|Director|Engineer|Developer|Analyst|Specialist)\b', line, re.IGNORECASE):
                if current_job:
                    experience.append(current_job)
                current_job = {'title': line.strip(), 'description': []}
            elif current_job and line.strip():
                current_job['description'].append(line.strip())
        
        if current_job:
            experience.append(current_job)
        
        return experience

class UIComponents:
    """Class to handle UI components and layout"""
    
    @staticmethod
    def sidebar():
        """Create sidebar content"""
        with st.sidebar:
            st.title("Resume Analyzer Pro")
            st.markdown("""
            Upload your resume for comprehensive analysis and personalized suggestions.
            
            **Supported formats:** PDF, DOCX
            """)
            
            st.markdown("---")
            st.markdown("### How It Works")
            st.markdown("""
            1. Upload your resume file
            2. Get instant analysis
            3. Review suggestions
            4. Download detailed report
            """)
            
            st.markdown("---")
            st.markdown("### Tips for Better Results")
            st.markdown("""
            - Use clear section headings
            - Include relevant keywords
            - Keep it concise (300-600 words)
            - Highlight achievements
            - Proofread carefully
            """)
    
    @staticmethod
    def display_score(score: int):
        """Display score with appropriate styling"""
        if score >= 80:
            st.success(f"Excellent Score: {score}/100")
        elif score >= 60:
            st.warning(f"Good Score: {score}/100")
        else:
            st.error(f"Needs Improvement: {score}/100")
        
        st.progress(score / 100)
    
    @staticmethod
    def create_download_link(content: str, filename: str) -> str:
        """Create a download link for the report"""
        b64 = base64.b64encode(content.encode()).decode()
        return f'<a href="data:file/markdown;base64,{b64}" download="{filename}">Download Report</a>'

def main():
    """Main application function"""
    UIComponents.sidebar()
    
    st.title("Resume Analyzer Pro")
    st.markdown("Upload your resume for comprehensive analysis and improvement suggestions")
    
    # File uploader with expanded options
    uploaded_file = st.file_uploader(
        "Choose your resume file",
        type=["pdf", "docx"],
        help="Maximum file size: 200MB",
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        # Process the resume
        with st.spinner("Processing your resume..."):
            resume_text = ResumeParser.extract_text(uploaded_file)
            
            if not resume_text.strip():
                st.error("Could not extract text from the uploaded file. Please try another file.")
                return
            
            # Perform analysis
            contact_info = ResumeAnalyzer.extract_contact_info(resume_text)
            sections = ResumeAnalyzer.analyze_structure(resume_text)
            skills = ResumeAnalyzer.extract_skills(resume_text)
            score, scoring_details = ResumeAnalyzer.calculate_score(resume_text, contact_info, sections, skills)
            suggestions = ResumeAnalyzer.generate_suggestions(resume_text, contact_info, sections, skills, score)
            experience = ResumeAnalyzer.extract_experience(resume_text)
        
        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Detailed Analysis", "Suggestions", "Full Text"])
        
        with tab1:
            st.subheader("Resume Overview")
            
            col1, col2 = st.columns([3, 2])
            with col1:
                UIComponents.display_score(score)
                
                # Quick stats
                st.markdown("### Key Metrics")
                cols = st.columns(3)
                cols[0].metric("Word Count", len(resume_text.split()))
                cols[1].metric("Skills Found", sum(len(skills_list) for skills_list in skills.values()))
                cols[2].metric("Sections", sum(sections.values()))
                
            with col2:
                st.markdown("### Contact Information")
                if any(contact_info.values()):
                    if contact_info['emails']:
                        st.markdown(f"**Email:** {', '.join(contact_info['emails'])}")
                    if contact_info['phones']:
                        st.markdown(f"**Phone:** {', '.join(contact_info['phones'])}")
                    if contact_info['linkedin']:
                        st.markdown(f"**LinkedIn:** {', '.join(contact_info['linkedin'])}")
                    if contact_info['github']:
                        st.markdown(f"**GitHub:** {', '.join(contact_info['github'])}")
                else:
                    st.warning("No contact information detected")
        
        with tab2:
            st.subheader("Detailed Analysis")
            
            # Scoring breakdown
            st.markdown("### Scoring Breakdown")
            for category, score_text in scoring_details.items():
                st.markdown(f"- **{category}:** {score_text}")
            
            # Skills analysis
            st.markdown("### Skills Analysis")
            if skills:
                for category, skills_list in skills.items():
                    with st.expander(f"{category} ({len(skills_list)} skills)"):
                        st.write(", ".join(skills_list))
            else:
                st.warning("No skills detected")
            
            # Experience analysis
            st.markdown("### Work Experience")
            if experience:
                for job in experience:
                    with st.expander(job['title']):
                        for desc in job['description']:
                            st.write(f"- {desc}")
            else:
                st.warning("No work experience detected")
        
        with tab3:
            st.subheader("Improvement Suggestions")
            
            # Categorize suggestions
            critical = [s for s in suggestions if s['type'] == 'critical']
            warnings = [s for s in suggestions if s['type'] == 'warning']
            info = [s for s in suggestions if s['type'] == 'info']
            success = [s for s in suggestions if s['type'] == 'success']
            
            if critical:
                st.markdown("#### Critical Issues")
                for suggestion in critical:
                    st.error(f"**{suggestion['title']}:** {suggestion['message']}")
            
            if warnings:
                st.markdown("#### Important Improvements")
                for suggestion in warnings:
                    st.warning(f"**{suggestion['title']}:** {suggestion['message']}")
            
            if info:
                st.markdown("#### Enhancement Suggestions")
                for suggestion in info:
                    st.info(f"**{suggestion['title']}:** {suggestion['message']}")
            
            if success:
                st.markdown("#### Strengths")
                for suggestion in success:
                    st.success(f"**{suggestion['title']}:** {suggestion['message']}")
        
        with tab4:
            st.subheader("Extracted Resume Text")
            with st.expander("View full text"):
                st.text(resume_text)
        
        # Generate and download report
        st.markdown("---")
        st.subheader("Generate Report")
        
        if st.button("Create Comprehensive Analysis Report"):
            report = f"""
# Resume Analysis Report
Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Score: {score}/100

## Scoring Breakdown:
{chr(10).join([f"- {cat}: {score}" for cat, score in scoring_details.items()])}

## Contact Information:
- Email: {', '.join(contact_info['emails']) if contact_info['emails'] else 'Not found'}
- Phone: {', '.join(contact_info['phones']) if contact_info['phones'] else 'Not found'}
- LinkedIn: {', '.join(contact_info['linkedin']) if contact_info['linkedin'] else 'Not found'}
- GitHub: {', '.join(contact_info['github']) if contact_info['github'] else 'Not found'}

## Skills Found:
{chr(10).join([f"### {cat}:{chr(10)}{chr(10).join([f'- {skill}' for skill in skills_list])}" for cat, skills_list in skills.items()])}

## Work Experience:
{chr(10).join([f"### {job['title']}{chr(10)}{chr(10).join([f'- {desc}' for desc in job['description']])}" for job in experience]) if experience else 'No work experience detected'}

## Suggestions:
{chr(10).join([f"- **{s['title']}**: {s['message']}" for s in suggestions])}

## Extracted Text:
{resume_text}
            """
            
            st.markdown(UIComponents.create_download_link(
                report,
                f"resume_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            ), unsafe_allow_html=True)

if __name__ == "__main__":
    if nlp:
        main()
    else:
        st.error("Cannot load required NLP model. Please install spaCy English model.")