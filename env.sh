conda create -n video_as_prompt python=3.10 -y
conda activate video_as_prompt

pip install -r requirements.txt

pip install -e ./diffusers

conda install -c conda-forge ffmpeg -y