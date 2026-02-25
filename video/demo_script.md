# DermScreen AI — Demo Video Script

**Target Length:** 2:45 - 3:00  
**Pacing:** Confident, clear, slightly energetic.  
**Visuals Note:** Ensure screen recording is clean (hide bookmarks, clear desktop).

---

### [0:00 – 0:20] Hook: The Problem
*(Visual: Split screen. Left side shows a map of rural health deserts. Right side shows a stressed community health worker looking at a skin lesion on a tablet.)*

**Voiceover:** "Over a billion people live in areas with zero access to dermatologists. For them, Community Health Workers are the only lifeline. But when a health worker sees a suspicious skin lesion, they face an impossible choice: send the patient on an expensive, multi-day journey to a clinic, or risk sending them home with an aggressive cancer."

### [0:20 – 0:40] Solution Intro
*(Visual: Transition to the DermScreen AI Gradio app cleanly loaded. Show the title and the "Offline Capable" badge if added, or just text highlighting it.)*

**Voiceover:** "Meet DermScreen AI. Powered by Google's MedGemma, it's an offline-capable, three-stage triage assistant that thinks like a dermatologist. It doesn't just classify an image; it works *with* the health worker to make a safe referral decision."

### [0:40 – 1:30] Live Demo Walkthrough
*(Visual: Cursor clicks 'Upload', selects a lesion image. Types "Patient complains of itching" in the CHW notes. Clicks "Assess".)*

**Voiceover:** "Let's look at it in action. In Stage One, the health worker snaps a photo. MedGemma analyzes the lesion, identifying key visual features like irregular borders."

*(Visual: Stage 2 questions magically appear on screen.)*

**Voiceover:** "But vision isn't enough. In Stage Two, MedGemma dynamically generates three targeted follow-up questions based specifically on what it just saw. The health worker asks the patient: 'Has it changed shape recently?' 'Does it bleed?' They type in the answers."

*(Visual: Typing in quick 'Yes', 'No', 'A little' answers. Clicks "Generate Triage".)*

**Voiceover:** "Stage Three brings it together. MedGemma synthesizes the visual data and the patient history to output a structured referral note and a clear triage color code—in this case, Red for Urgent Referral. This note can be immediately WhatsApped to the regional clinic."

### [1:30 – 2:00] Technical Depth
*(Visual: Switch tab to show `model_comparison.png` bar chart showing Zero-shot vs LoRA.)*

**Voiceover:** "Why MedGemma? General vision models lack deep medical alignment. By taking MedGemma-4b and applying parameter-efficient LoRA fine-tuning on the ISIC dataset, we adapted it to standard dermatology taxonomy. As you can see, our fine-tuning yielded a massive 13% improvement in accuracy over the zero-shot baseline, all while training efficiently on a single GPU."

### [2:00 – 2:20] Deployment & Hardware
*(Visual: Switch to terminal showing the model running locally, maybe showing the 4-bit loading logs.)*

**Voiceover:** "Crucially, rural health workers don't have reliable internet. Because DermScreen uses 4-bit quantization, the entire pipeline runs entirely locally on a standard laptop or edge device. No cloud required, ensuring patient data privacy and total uptime."

### [2:20 – 2:45] Impact & Close
*(Visual: Cut back to the successful referral note UI. Fade to DermScreen AI Logo and GitHub Repo link.)*

**Voiceover:** "By empowering health workers with expert-level decision support, DermScreen AI reduces false referrals, saves clinic resources, and catches critical melanomas early. Thank you for watching, and we're excited to see how tools like MedGemma can revolutionize global health equity."

---

### 🎥 Recording Tips (Read Before Filming)
- **Software:** Use OBS Studio for screen recording.
- **Resolution:** 1080p (1920x1080) at 60fps.
- **Audio:** Use a lapel mic or a good USB mic (Blue Yeti or equivalent). Do not use laptop built-in audio.
- **Accessibility:** Add burnt-in captions or provide a clean `.srt` file when uploading to YouTube/Kaggle.
- **Clean Environment:** Turn off notifications (Do Not Disturb). Hide the Windows taskbar or Mac Dock.
