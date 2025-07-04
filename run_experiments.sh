#!/bin/bash

model="gpt"
temperatures=("0.00" "0.25" "0.50")
prompting_techniques=("basic" "few-shot" "react+cot" "grounded-citing" "all")

for temperature in "${temperatures[@]}"; do
  for tech in "${prompting_techniques[@]}"; do
    echo "Running with model=$model, temperature=$temperature, prompting_technique=$tech"
    
    python main.py content/RINF_Application_guide_V1.6.1.html \
      --model "$model" \
      --temperature "$temperature" \
      --prompting_technique "$tech"
    
    output_file="output/rinf-application-guide-v1-6-1_${model}_${temperature}_${tech}.ttl"
    
    echo "Validating $output_file"
    python validation/validation.py "$output_file"
    
  done
done