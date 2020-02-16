eval "$(conda shell.bash hook)"
conda activate modifiers

for filename in ../data/metamap/inputs/*.txt; do
    outputfile="../data/metamap/outputs/$(basename "$filename" .txt).txt"
    python mmlrestclient.py \
        https://ii-public2.nlm.nih.gov/metamaplite/rest/annotate \
        $filename \
        --output $outputfile \
        --docformat sldiwi \
        --resultformat mmi
done
