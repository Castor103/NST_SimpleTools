#!/bin/bash
# ref : https://the-dev.tistory.com/14

find ./target_files > files_list.txt

while read filePath; do
    python3 checkHKFileData.py -f $filePath -s

done < files_list.txt