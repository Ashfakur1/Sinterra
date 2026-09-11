# নতুন ওয়ার্কফ্লো: রিয়েল ল্যাব ব্যাচ vs সিনথেটিক ব্যাচ

## যা বদলেছে

আগে `generate_dataset.py` (Script 1)-এর ভেতরে ৮টা ল্যাব ব্যাচ সরাসরি Python কোডে
হার্ডকোড করা ছিল (`_LAB_RAW` variable)। এখন সেটা সরিয়ে একটা আলাদা ফাইলে নেওয়া
হয়েছে:

- **`lab_batches_raw.csv`** → আপনার রিয়েল ল্যাব ব্যাচগুলো থাকবে এখানে। এই
  ফাইলটা **`generate_dataset.py`, `inverse_design.py`, `train_forward_model.py`,
  `streamlit_app.py` যে ফোল্ডারে আছে ঠিক সেই একই ফোল্ডারে** রাখবেন (কোনো `data/`
  সাবফোল্ডারে না)। নতুন টেস্ট করলে এই ফাইলেই একটা নতুন row যোগ করবেন।
- **`generate_dataset.py`** → এখন প্রতিবার রান করার সময় এই CSV পড়ে, তার উপর
  ভিত্তি করে Ridge coefficient re-fit করে, LOO-CV চালায়, এবং ১০০০টা সিনথেটিক
  ব্যাচ জেনারেট করে — সবকিছু আপনার latest ল্যাব ডেটার সাথে সামঞ্জস্যপূর্ণ থাকে।
- **`streamlit_app.py`** → এখন `dataset.csv` ও `metadata.json` ফাইলের
  modification time cache key হিসেবে ব্যবহার করে, তাই `generate_dataset.py`
  রি-রান করার সাথে সাথেই app নিজে থেকে নতুন ডেটা ধরে নেয় — **app restart বা
  "Clear cache" করার দরকার নেই।** শুধু ব্রাউজারে rerun/refresh হলেই (Streamlit
  widget-এ কিছু একটা টাচ করলে বা "Rerun" চাপলে) নতুন MOR/WA/Shrinkage
  min-max limit দেখাবে।

## নতুন টেস্ট করলে কী করবেন

1. `lab_batches_raw.csv` ফাইলটা (যেটা আপনার scripts-এর ফোল্ডারেই আছে)
   Excel/Notepad-এ খুলুন।
2. নিচের কলামগুলোতে আপনার নতুন ব্যাচের ডেটা বসিয়ে একটা নতুন row যোগ করুন:

   | কলাম | মানে |
   |---|---|
   | AG98, AG22, AG23, SodaF, PotashF, Crushing, ETP, NaSil | কাঁচামালের wt% (Sigma=100 না হলেও চলবে, কোড নিজেই normalize করে) |
   | MOR_kgf_mm2 | ফ্লেক্সারাল স্ট্রেংথ, kgf/mm² (টেস্টার থেকে যেভাবে পান সেভাবে) |
   | WA_fraction | ওয়াটার অ্যাবজর্পশন, fraction হিসেবে (যেমন 3.59% হলে লিখবেন 0.0359) |
   | Shrinkage_pct | ফায়ার্ড শ্রিংকেজ, % |

3. সেভ করে `generate_dataset.py` রান করুন:
   ```
   python generate_dataset.py
   ```
4. এতেই `data/dataset.csv`, `data/lab_batches.csv`, `data/metadata.json`
   ইত্যাদি সব নতুন ব্যাচ ধরে re-generate হয়ে যাবে।
5. যদি `streamlit_app.py` আগে থেকেই চালু থাকে, তাহলে শুধু ব্রাউজারে গিয়ে যেকোনো
   widget-এ ক্লিক করুন বা "Rerun" চাপুন — নতুন `dataset.csv` নিজে থেকেই লোড
   হয়ে যাবে, app restart করার দরকার নেই।

## অন্য স্ক্রিপ্টগুলোর কী হবে (Script 2, 3, 4, reliability_analysis.py)

**এগুলোতে কোনো পরিবর্তন লাগেনি — সবই আগের মতোই কাজ করবে।** কারণ:

- Script 2, 3, 4 শুধু `data/dataset.csv`, `data/metadata.json`,
  `data/feature_cols.json` ফাইল থেকে ডেটা পড়ে — Script 1-এর ভেতরের
  `_LAB_RAW` variable-এ সরাসরি হাত দেয় না। Script 1-এর আউটপুট ফরম্যাট
  অপরিবর্তিত রাখা হয়েছে, তাই ডাউনস্ট্রিমে কিছুই ভাঙবে না।
- `reliability_analysis.py`, `generate_dataset`-কে module হিসেবে import করে
  (`import generate_dataset as gd`) এবং `gd._LAB_RAW` ব্যবহার করে। এই
  variable আগে হার্ডকোড থেকে আসতো, এখন CSV থেকে আসে — কিন্তু ফলাফল (list of
  dict) একই গঠনের, তাই সেটাও অপরিবর্তিত অবস্থায় কাজ করবে।

তাই পুরো পাইপলাইনের একমাত্র "single source of truth" এখন `lab_batches_raw.csv`
— সেটা আপডেট করলেই বাকি সব (synthetic data, models, feature importance,
inverse design recommendation, reliability plot) automatically তার সাথে
সামঞ্জস্যপূর্ণভাবে regenerate হবে।

## ভ্যালিডেশন যোগ করা হয়েছে

- CSV ফাইল না থাকলে বা কলাম মিসিং থাকলে স্পষ্ট error message দেখাবে।
- ৪টার কম ব্যাচ থাকলে error দেবে (Ridge fit অর্থবহ হওয়ার জন্য ন্যূনতম প্রয়োজন)।
- সব জায়গায় যেখানে ব্যাচ-সংখ্যা "8" হার্ডকোড ছিল (log message, প্লট লেবেল,
  metadata.json-এর `generation_method`/`validation_method`) সেগুলো এখন
  dynamic — নতুন ব্যাচ যোগ করলে এগুলো নিজে থেকেই "9", "10" ইত্যাদি দেখাবে।

## Folder structure (এখন যেমন হওয়া উচিত)

```
your_project_folder/
├── generate_dataset.py
├── inverse_design.py
├── train_forward_model.py
├── streamlit_app.py
├── lab_batches_raw.csv        ← এটাও এখানেই, data/ এর ভেতরে না
└── data/                      ← generate_dataset.py এটা নিজে থেকে বানায়
    ├── dataset.csv
    ├── lab_batches.csv
    ├── metadata.json
    └── ...
```

## Streamlit app auto-reload (নতুন)

`streamlit_app.py`-তে `_load_dataset()` ও `_load_meta()` ফাংশন দুটোতে এখন
`dataset.csv` / `metadata.json`-এর file-modification-time একটা argument
হিসেবে পাস করা হয় (`@st.cache_data` decorator সহ)। ফলে:

- ফাইল অপরিবর্তিত থাকলে → cache থেকেই দ্রুত লোড হয় (অপ্রয়োজনীয় re-read হয় না)।
- ফাইল বদলে গেলে (mtime বদলায়) → cache নিজে থেকেই invalidate হয়ে ফাইল আবার
  পড়ে নেয়।

তাই `generate_dataset.py` রি-রান করার পর app-এ শুধু একটা rerun/refresh
(widget touch বা Streamlit-এর "Rerun" বাটন) হলেই MOR/WA/Shrinkage-এর
min-max limit স্বয়ংক্রিয়ভাবে নতুন ডেটা অনুযায়ী আপডেট হয়ে যাবে — পুরো app
বন্ধ করে আবার চালানোর দরকার নেই।

## `train_forward_model.py`-তেও একটা জায়গা ঠিক করা হয়েছে

`Train: 800 synthetic | Test: 200 synthetic + 8 experimental` — এই লাইনটা
আগে থেকেই dynamic ছিল (`len(lab_idx)` ব্যবহার করে, `df["source"]=="lab_batch"`
থেকে গণনা করে), তাই ব্যাচ ১১টা করলে এটা নিজে থেকেই
`... + 11 experimental` দেখাবে — কোনো পরিবর্তন লাগেনি।

তবে এর ঠিক নিচে আরেকটা লাইন ছিল যেটা হার্ডকোড করা ছিল:
```
Experimental-batch-only metrics (n=8, held-out, R² not reported ...)
```
এটা ঠিক করে দিয়েছি — এখন `n={int(lab_mask_test.sum())}` দিয়ে actual held-out
ব্যাচ সংখ্যা বসে। ১১টা ব্যাচ দিয়ে টেস্ট করে দেখা হয়েছে — আউটপুটে ঠিকই
`Test: 800 synthetic | Test: 200 synthetic + 11 experimental` এবং
`Experimental-batch-only metrics (n=11, ...)` দেখাচ্ছে।

(নোট: module docstring/comment-এর ভেতরে "8 experimental batches", "n=8"
টেক্সট এখনও আছে — এগুলো শুধু ডকুমেন্টেশন, কোডের আউটপুটে প্রভাব ফেলে না,
তাই স্পর্শ করিনি।)

## যাচাই করা হয়েছে

মূল ৮টা ব্যাচ দিয়ে রান করে আউটপুট (LOO-CV MAE, dataset row count, property
range/mean/CV) হুবহু আপনার আগের রান-লগের সাথে মিলিয়ে দেখা হয়েছে — সংখ্যায়
কোনো পার্থক্য নেই। একটা টেস্ট ৯ম ব্যাচ যোগ করেও চালিয়ে দেখা হয়েছে —
coefficient re-fit, dataset (lab=9), এবং সব প্লট ঠিকভাবে আপডেট হয়েছে।
`lab_batches_raw.csv` স্ক্রিপ্টের নিজের ফোল্ডারে (data/ সাবফোল্ডার ছাড়া)
রেখেও রান করে দেখা হয়েছে — একই ফলাফল পাওয়া গেছে।
