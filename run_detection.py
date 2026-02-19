import json

def collect_number_contexts(input_path, words_before=20, words_after=20):
    contexts = []
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except FileNotFoundError:
        print(f"Input file {input_path} not found")
        return []
    except Exception as e:
        print(f"Read error: {e}")
        return []
    
    tokens = full_text.split()
    print(f"Read {len(tokens)} tokens from input file")
    
    for i, token in enumerate(tokens):
        cleaned = token.strip('.,;:!?()[]{}"\'')
        if not cleaned:
            continue
            
        try:
            float(cleaned)
        except ValueError:
            continue
            
        start = max(0, i - words_before)
        end = min(len(tokens), i + 1 + words_after)
        
        before = tokens[start:i]
        after = tokens[i + 1:end]
        
        context_parts = before + [cleaned] + after
        input_text = ' '.join(context_parts)
        
        entry = {
            'input_text': input_text,
            'number': cleaned,
            'label': None
        }
        contexts.append(entry)
    
    print(f"Collected {len(contexts)} number samples")
    return contexts


def save_contexts_to_json(contexts, output_path="number_contexts.json"):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(contexts, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(contexts)} entries to {output_path}")
    except Exception as e:
        print(f"Write error: {e}")


if __name__ == "__main__":
    contexts = collect_number_contexts("input.txt")
    
    if contexts:
        save_contexts_to_json(contexts)
        print("Data ready for sentence-transformers pipeline")
