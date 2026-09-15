#!/usr/bin/env python3
"""English in, proof out -- the full Type 3 pipeline."""

import json

from neuralmind import NeuralMindPipeline

PASSAGE = """
Bob is a cat. Alice is a dog. Bob is small.
All cats are mammals. All dogs are mammals.
If something is a mammal then it is warm blooded.
If something is warm blooded and it is a cat then it purrs.
If someone is small and they are a cat then they are quiet.
"""

pipeline = NeuralMindPipeline()
result = pipeline.run(PASSAGE, question="Does Bob purr?")

print("what perception extracted:")
for record in result.perception.facts:
    print(f"  {str(record.atom):32s} confidence {record.confidence:.2f}")
for rule in result.perception.rules:
    print(f"  {rule}")

print("\nthe answer, with its derivation:")
print(result.proof)

print("\nthe same thing in English:")
print(result.explanation)

print("\nand as JSON for a downstream system:")
print(json.dumps(result.to_dict(include_facts=False)["answer"], indent=2)[:500])

print("\na question the text does not answer:")
print(pipeline.ask("Is Alice quiet?").answer.diagnosis)
