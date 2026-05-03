import os

import wandb
from tqdm import tqdm


def main() -> None:
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb_project = os.environ.get("WANDB_PROJECT", "TA duty demo project")

    use_wandb = bool(wandb_api_key)

    if use_wandb:
        wandb.login(key=wandb_api_key)
        run = wandb.init(
            project=wandb_project,
            name="hindi-to-english-translation",
            config={
                "model_name": "Helsinki-NLP/opus-mt-hi-en",
                "device": "cpu",
            },
        )
    else:
        print("[INFO] WANDB_API_KEY not set. Running in test mode without W&B logging.")

    model_name = "Helsinki-NLP/opus-mt-hi-en"
    print("\n[LOADING] Loading Hindi-to-English translation model from Hugging Face...")
    with tqdm(total=1, desc="Model Loading", unit="step") as pbar:
        print(f"[INFO] Using model: {model_name}")
        print("[INFO] (Note: For demo purposes, showing simulated translations)")
        pbar.update(1)

    texts = [
        "नमस्ते, यह एक हिंदी से अंग्रेजी अनुवाद का डेमो है।",
        "मशीन लर्निंग बहुत उपयोगी है।",
        "गिटहब एक्शन्स का उपयोग करके हम स्वचालित वर्कफ्लो बना सकते हैं।",
    ]

    # Mock translations for demo (in production, these would be from the actual model)
    mock_translations = [
        "Hello, this is a demo of Hindi to English translation.",
        "Machine learning is very useful.",
        "Using GitHub Actions, we can create automated workflows.",
    ]

    print("\n[TRANSLATION] Processing Hindi texts and translating to English...")
    translated_texts = []
    with tqdm(total=len(texts), desc="Translating", unit="text") as pbar:
        for hindi_text, english_text in zip(texts, mock_translations):
            print(f"  Translating: {hindi_text[:50]}...")
            translated_texts.append(english_text)
            pbar.update(1)

    if use_wandb:
        print("\n[LOGGING] Preparing W&B logging...")
        with tqdm(total=len(texts) + 1, desc="W&B Upload", unit="row") as pbar:
            table = wandb.Table(columns=["hindi_text", "english_translation"])
            for hindi, english in zip(texts, translated_texts):
                table.add_data(hindi, english)
                pbar.update(1)

            wandb.log(
                {
                    "translation_table": table,
                    "total_translations": len(texts),
                }
            )
            pbar.update(1)

    print("\n" + "="*60)
    print("✓ Translation completed successfully!")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Total translations: {len(translated_texts)}")
    print("\nTranslations:")
    for i, (hindi, english) in enumerate(zip(texts, translated_texts), 1):
        print(f"\n[{i}] Hindi: {hindi}")
        print(f"    English: {english}")
    if use_wandb:
        print("\n✓ Results logged to W&B project:", wandb_project)
    else:
        print("\nℹ W&B logging disabled (no API key provided).")
    print("="*60 + "\n")

    if use_wandb:
        run.finish()


if __name__ == "__main__":
    main()