# We load the normalized and oriented images and plot them to check the registration
FOLDER=./data/normalized/nyul_normalized_oriented/
FILENAME=abbytoernie_nyul

python ./scripts/5-utils/slicer.py \
--image ${FOLDER}/${FILENAME}.nii \
--axis 0 --pos 125 --bound 50 200 30 210 --output ${FOLDER}/${FILENAME}.png

for i in {1..4}
do
    FOLDER=outputs/my_registration_${i}/norm_registered/
    FILENAME=CurrentState
    python ./scripts/5-utils/slicer.py \
    --image ${FOLDER}/${FILENAME}.mgz \
    --axis 0 --pos 125 --bound 50 200 30 210 --output ${FOLDER}/${FILENAME}.png
done