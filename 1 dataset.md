# Dataset Folder structure:


Look my audio speech dataset is format is as such:




This is the folder structure

```
data/
├── BPAC/
│   ├── chattogram/
│   ├── dhaka/
│   ├── Faridpur/
│   ├── khulna/
│   ├── mymensingh/
│   ├── rajshahi/
│   ├── rangpur/
│   └── sylhet/
└── best_diverse_sentences.csv
```

audio files are located as such:

For male:
```
data/BPAC/region/speaker_male_no_name/sound_001.wav
...
...
...
data/BPAC/region/speaker_male_no_name/sound_250.wav
```

For female:
```
data/BPAC/region/speaker_female_no_name/sound_001.wav
...
...
...
data/BPAC/region/speaker_female_no_name/sound_250.wav
```


For more context:

```
data/
├── BPAC/
│   ├── chattogram/
│   │   ├── speaker_female_001_mahi/
│   │   │   ├── sound_001.wav
│   │   │   ├── sound_002.wav
│   │   │   ├── sound_003.wav
│   │   │   ├── sound_004.wav
│   │   │   ├── sound_005.wav
│   │   │   ├── sound_006.wav
│   │   │   ├── sound_007.wav
│   │   │   └── ...
│   ├── dhaka/
│   ├── Faridpur/
│   ├── khulna/
│   ├── mymensingh/
│   ├── rajshahi/
│   ├── rangpur/
│   └── sylhet/
└── best_diverse_sentences.csv
```

```
data/
├── BPAC/
│   ├── <district_name>/
│   │   ├── speaker_<gender>_<speaker_id>_<speaker_name>/
│   │   │   ├── sound_001.wav
│   │   │   ├── sound_002.wav
│   │   │   ├── ...
│   │   │   └── sound_NNN.wav
│   │   ├── speaker_<gender>_<speaker_id>_<speaker_name>/
│   │   └── ...
│   ├── <district_name>/
│   └── ...
└── best_diverse_sentences.csv
```

Look `gender` `speaker_id` are combinely unique for that region. however dont consider them unique for the entire dataset, 

```
data/
├── BPAC/
│   ├── chattogram/
│   │   ├── speaker_female_001_mahi/
│   │   ├── speaker_female_002_afia/
│   │   ├── speaker_female_003_fida/
│   │   ├── speaker_male_001_riajul/
│   │   ├── speaker_male_002_mamun/
│   │   └── speaker_male_003_faiaz/

```
As you can see here, female with id `001`, and male with `001` exists.

Also for dhaka
```
THESIS-BPAC-EXPLORATORY-DATA-ANALYSIS-OF
│
├── data
│   └── BPAC
│       ├── chattogram
│       ├── dhaka
│       │   ├── speaker_female_001_shakila
│       │   ├── speaker_female_002_tasnim
│       │   ├── speaker_female_003_anindita
│       │   ├── speaker_male_001_riham
│       │   ├── speaker_male_002_turab
│       │   └── speaker_male_003_shifat
│       └── ...
```
As you can see id repeats. 

So you also take region name in consideration. 

To uniquely identify or name a writer, you make it like
```
<region>_<gender>_<id>_<name> 
```





Now about the dataset.

The text is located in the csv file `best_diverse_sentences.csv`

You can get example format like below,
```csv
id,sentence
1,গত শুক্রবার তাঁরা প্রায় এক ঘণ্টা ধরে হোয়াইট হাউসে বৈঠক করেছেন।
2,ভারতের হয়ে একাই চার উইকেট শিকার করেছেন কুলদীপ যাদব।
3,শুরুতে শুধু ফোনে ভিডিও করতেন।
4,এ তথ্য বিজ্ঞানীদের বলতে সাহায্য করে ভূমিকম্পের উৎপত্তি কোথায় হয়েছে।
5,কার্বন নির্গতকারী বিশ্বের প্রথম রেইনফরেস্ট বলা হচ্ছে অস্ট্রেলিয়ার ক্রান্তীয় রেইনফরেস্টকে।
6,ক্যাম্ফার তো বটেই স্বাগতিক বোলারদের মাথা ব্যথার কারণ হয়ে দাঁড়ান লেগস্পিনার হোয়ে।
7,বর্তমানে ভিসা সমস্যাসহ নানা জটিলতার কারণে বাংলাদেশি নাবিকদের জন্য বিদেশি জাহাজে কাজ পাওয়ার সুযোগ ক্রমেই সংকুচিত হয়ে যাচ্ছে।
8,সোমবার স্বাস্থ্য অধিদপ্তরের হেলথ ইমার্জেন্সি অপারেশন সেন্টার ও কন্ট্রোলরুম থেকে পাঠানো ডেঙ্গু বিষয়ক নিয়মিত সংবাদ বিজ্ঞপ্তিতে এ তথ্য জানানো হয়।
9,"টানা চার মাস, আগস্ট থেকে নভেম্বর পর্যন্ত চলে এ কর্মযজ্ঞ।"
10,ডান দিকে ঝাঁপিয়ে তার স্পটকিক রুখে দেন মেনিয়।
```


