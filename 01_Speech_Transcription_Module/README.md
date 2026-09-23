# Speech Transcription Module

Орысша аудионы Whisper large-v3 моделімен тануға арналған модуль.

## Қазіргі күйі

`requirements.txt` арқылы faster-whisper орнатуға болады. Тәуелділіктер GitHub серверіне емес, бағдарламаны іске қосатын компьютердің виртуалды ортасына орнатылады. Толық `transcribe.py` CLI және әр аудиоға жеке TXT/JSON экспорттау осы өзгеріске кірмейді.

## Орнату: Windows PowerShell

Python 3.11 (64-bit) пайдалануды ұсынамыз. Репозиторийді жүктегеннен кейін оның түбірінен:

```powershell
cd 01_Speech_Transcription_Module
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "from faster_whisper import WhisperModel; print('faster-whisper import OK')"
```

Виртуалды ортаны белсендіру міндетті емес: командалар оның Python файлын тікелей пайдаланады. Импортты тексеру модельді жүктемейді.

Linux/macOS:

```bash
cd 01_Speech_Transcription_Module
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

## Модельді пайдалану үлгісі

Төмендегі Python кодын модуль папкасынан орындаңыз. Алғаш рет интернет қажет: large-v3 салмақтары Hugging Face кэшіне жүктеледі. Жүктеу көлемі бірнеше ГБ болуы мүмкін. Кэштегі салмақтарды Git-ке қоспаңыз.

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cpu", compute_type="int8")
segments, info = model.transcribe(
    "01_datasets_audio/01_audio1.mp3",
    language="ru",
    task="transcribe",
    beam_size=5,
    vad_filter=True,
    word_timestamps=True,
)
for segment in segments:
    print(f"[{segment.start:.2f}–{segment.end:.2f}] {segment.text.strip()}")
```

Бұл үлгі нәтижені терминалға шығарады; TXT/JSON файлдарын жасамайды.
`segments` — генератор: толық транскрипция үшін оны соңына дейін оқу қажет.
CPU арқылы large-v3 баяу жұмыс істеуі мүмкін.

## NVIDIA GPU

GPU үшін сәйкес NVIDIA драйвері, CUDA 12 cuBLAS және cuDNN 9 қажет. Кітапханаларды орнату жөніндегі [ресми нұсқаулықты](https://github.com/SYSTRAN/faster-whisper#gpu) қараңыз. Windows-та қажетті DLL файлдары PATH арқылы табылуы тиіс.

GPU дайын болса, үлгідегі модельді жүктеу жолын ауыстырыңыз:

```python
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
```

CUDA/DLL қатесі болса, ортаны дұрыстаңыз немесе CPU нұсқасын пайдаланыңыз.
Жад жеткіліксіз болса, GPU-да `compute_type="int8_float16"` қолданып көріңіз немесе CPU-ға ауысыңыз.
Модель жүктеу қатесінде интернет пен дискідегі бос орынды тексеріңіз.
Аудионы оқу қатесінде файл жолын және файлдың ашылатынын тексеріңіз.

## Келісілген нәтиже құрылымы

Болашақ транскрипция скрипті әр аудиоға жеке папка жасауы тиіс:

```text
outputs/
├── 01_audio1/
│   ├── 01_audio1.txt
│   └── 01_audio1.json
└── 01_audio2/
    ├── 01_audio2.txt
    └── 01_audio2.json
```

Бұл — күтілетін құрылым; дайын транскрипциялар репозиторийге қосылған жоқ.
Эталон мәтінсіз WER/CER есептелмейді.

## Тексеру шегі

Тәуелділік пен пайдалану үлгісі faster-whisper ресми құжаттамасына сәйкес берілген. Осы өзгерісті дайындау кезінде кітапхананы нақты ортаға орнату және аудионы модельмен өңдеу орындалған жоқ.
