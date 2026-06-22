#############################################################################################
#                                                                                           # 
#                                The Third DISPLACE Challenge                               #
#              DISPLACE-M - DIarization and Speech Processing for LAnguage                  # 
#                 understanding in Conversational Environments - Medical                    #
#                                                                                           #
#                           https://displace2026.github.io/                                 #
#                                                                                           #  
#############################################################################################

1. About
Inspired by the previous session of DISPLACE challenge, we have launched the third DISPLACE-M 
challenge (https://displace2026.github.io/). The DISPLACE-M challenge provides a unique dataset 
of medical conversations between Community health workers and local residents in two Indian 
languages, Hindi and Kannada, collected in a wide geographic region covering different dialects. 
The dataset presents unique challenges such as spontaneous dialogue, foreground speech overlap, 
background speech, dialectal variation, and environmental noise in rural healthcare settings, 
making it an unprecedented resource for advancing low-resource, multi-dialect conversational AI. 

Here we provide the development dataset for Hindi to be used in the challenge.

2. Directory structure
```text
├── README.md
├── LICENSE.md
└── Hindi
    ├── Audio
    │   └── <Record_ID>.wav
    └── GT
        └── gt_data_summarization.csv
```

3. Directory contents
    Audio : Contains audio files (*.wav) for different sessions
    GT    : Contains Record ID of audio files and corresponding ground truth summary

    LICENSE.md  : Contains the License information

4. Audio file description
Each audio file corresponds to a unique conversation. The audio file details are as follows: 
	- Sampling Rate : 16 kHz
	- Channels      : 1	
	- Format        : wav

5. GT file description
    Record ID : ID of the audio file
    Summary   : Ground truth summary of the audio file

    ## Sample GT Data Format
    ```csv
    rec_id,gt_summary
    2006763,"Patient reports persistent cough and mild fever for three days. Doctor advises rest, increased fluid intake, and prescribes cough syrup."
    2006764,"Patient discusses recurring headaches and eye strain. Doctor recommends reducing screen time and schedules a follow-up vision test."
    ```

6. Instructions
-   Adhere to the Terms and Conditions document signed by your team.
-   This dataset is released for use in the DISPLACE Challenge only.
-   This dataset should not be shared with non-participants of this challenge. 


7. Contact Us
Please reach out to displace2026@gmail.com for any queries.

8. Organizers
    Team DISPLACE
    - Prof. Sriram Ganapathy | Associate Professor, LEAP Lab, IISc, Bangalore, India 
    - Dr Dhanya E | Postdoctoral Researcher, LEAP Lab, IISc, Bangalore, India
    - Noumida A | Postdoctoral Researcher, LEAP Lab, IISc, Bangalore, India
    - Ankita Meena | M.Tech Student, IISc Bangalore, India
    - Victor Azad | M.Tech Student, IISc Bangalore, India
    - Manas Sameer Nanivadekar | Research Intern, IISc Bangalore, India
    - Prof. Deepu Vijayasenan | Associate Professor, NITK, Surathkal, India
    - Pratik Roy Chowdhuri | Research Scholar, NITK, Surathkal, India
    - Ashwini Nagaraj Shenoy | Junior Research Fellow, NITK Surathkal, India
    - Supreeth A | Junior Research Fellow, NITK Surathkal, India
    - Dr. Kalluri Shareef Babu | Assistant Professor, UPES Dehradun, India
    - Dr. Srikanth Raj Chetupalli | Assistant Professor, IIT Bombay, India


Indian Institute of Science (IISc), Bangalore-560012, India
National Institute of Technology Karnataka (NITK), Surathkal-575025, India
UPES Dehradun, Uttarakhand-248007, India
Indian Institute of Technology Bombay (IITB), Mumbai-400076, India

#############################################################################################
