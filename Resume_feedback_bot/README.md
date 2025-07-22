# AI Resume Analyzer

A web app that allows users to upload their resumes (PDF or DOCX format), analyze them for key skills, sections, and grammar issues, and generate a score to help improve the quality of their resumes.

---

## ✨ Features

- 📄 **Resume Upload**: Users can upload their resumes in PDF or DOCX formats.
- 📊 **Resume Analysis**: The app analyzes the resume for word count, grammar issues, relevant skills, and standard sections (Education, Experience, Projects, etc.).
- 🧮 **Resume Score**: A score out of 100 is generated based on the content and quality of the resume.
- 🔍 **Skills Detection**: The app detects a wide range of technical and soft skills based on keywords.
- 📋 **Section Detection**: The app identifies if common sections like Education, Experience, and Skills are included in the resume.
- 💡 **Suggestions**: Provides suggestions to improve the resume based on detected errors and missing information.
- 🚀 **Interactive Dashboard**: The results are displayed in an easy-to-read format with progress bars and suggestions for improvement.

---

### 📝 Important Notes

- **File Types**: Currently, only PDF and DOCX formats are supported for upload.
- **Suggestions**: The app generates recommendations based on detected sections, skills, and grammar issues.
- **Future Updates**: Enhancements will be made for better grammar detection, additional features like parsing of more complex sections, and integration of a more robust scoring algorithm.

---

## 🛠️ Setup Instructions

Follow these steps to run the project on your machine:

### 1. 📦 Clone the Repository

```bash
git clone https://github.com/SwastikKaushal1/AI-RESUME-ANALYSER.git
cd AI-RESUME-ANALYSER
```

### 2. 📥 Install Requirements

Make sure you have Python installed (3.8+ recommended).

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 3. 🚀 Run the Project

Run the app using:

```bash
streamlit run app.py
```

---
## 📸 Screenshots

1. **Resume Upload Page**  
   ![](ss1.jpg<img width="1885" height="848" alt="image" src="https://github.com/user-attachments/assets/6ceeecbf-503a-490e-8c6b-83eba8922bdd" />
)

2. **Resume Preview & Analyze Button**  
   ![](ss2.jpg)

3. **Skills & Grammar Analysis**  
   ![](ss3.jpg)

4. **Section Detection & Score**  
   ![](ss4.jpg)

## 💡 Usage

**How it works:**
1. Upload a PDF or DOCX resume file.
2. The app analyzes the resume and generates a score based on word count, grammar, skills, and sections.
3. The app provides suggestions to improve your resume for better chances of getting noticed by employers.

---

## 🛠️ Technologies Used

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0-blue?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7-blue?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io/)

---

## 📸 Screenshots

1. **Resume Upload Page**  
   ![](ss1.jpg)

2. **Resume Preview & Analyze Button**  
   ![](ss2.jpg)

3. **Skills & Grammar Analysis**  
   ![](ss3.jpg)

4. **Section Detection & Score**  
   ![](ss4.jpg)

5. **Downloadable Report**  
   ![](ss5.jpg)

---

## 🚧 Roadmap

- **Enhancements in Grammar Detection**: Improve grammar checking capabilities.
- **Expanded Skills Set**: Add more skills and industry-specific terms.
- **Enhanced User Interface**: Make the dashboard more user-friendly and visually appealing.
