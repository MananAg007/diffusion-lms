# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
import itertools
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2TokenizerFast
from model.modeling_llada import LLaDAModelLM
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from utils.grammar_check import grammar_check, grammar_error_count
from nltk import word_tokenize, ngrams
from nltk.probability import FreqDist
from collections import defaultdict
import math

# Global variables for perplexity model (lazy initialization)
_perplexity_model = None
_perplexity_tokenizer = None

def get_perplexity_model(device='cuda'):
    """Lazy initialization of GPT-2 model for perplexity calculation."""
    global _perplexity_model, _perplexity_tokenizer
    if _perplexity_model is None:
        _perplexity_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        _perplexity_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
    return _perplexity_model, _perplexity_tokenizer

def calculate_perplexity(text, device='cuda'):
    """
    Calculate perplexity of text using GPT-2.
    Lower perplexity = better token compatibility/coherence.

    Args:
        text: Input text string
        device: Device to run model on

    Returns:
        perplexity: Float value (lower is better)
    """
    if not text or not text.strip():
        return float('inf')

    model, tokenizer = get_perplexity_model(device)

    # Tokenize
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)

    # Calculate perplexity
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    return perplexity

def calculate_ngram_score(text, n=3):
    """
    Calculate n-gram based compatibility score for text.
    Uses n-gram language model probability as a measure of coherence.
    
    The score is based on:
    1. Average log probability of n-grams in the text
    2. Smoothed using Laplace smoothing to handle unseen n-grams
    
    Higher score = better n-gram compatibility/coherence.
    
    Args:
        text: Input text string
        n: N-gram size (default=3 for trigrams)
    
    Returns:
        score: Float value (higher is better, typically negative)
    """
    if not text or not text.strip():
        return -float('inf')
    
    try:
        # Try to ensure punkt tokenizer is available
        try:
            import nltk
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                import nltk
                nltk.download('punkt_tab', quiet=True)
            except:
                # Fallback to simple whitespace tokenization if download fails
                tokens = text.lower().split()
                if len(tokens) < n:
                    return -10.0 * (n - len(tokens))
                
                n_grams = list(ngrams(tokens, n))
                context_grams = list(ngrams(tokens, n-1)) if n > 1 else None
                ngram_freq = FreqDist(n_grams)
                context_freq = FreqDist(context_grams) if context_grams else None
                vocab_size = len(set(tokens))
                log_prob_sum = 0.0
                
                if n == 1:
                    total_tokens = len(tokens)
                    for gram in n_grams:
                        count = ngram_freq[gram]
                        prob = (count + 1) / (total_tokens + vocab_size)
                        log_prob_sum += math.log(prob)
                else:
                    for i, gram in enumerate(n_grams):
                        context = context_grams[i] if context_grams else None
                        ngram_count = ngram_freq[gram]
                        context_count = context_freq[context] if context else len(tokens)
                        prob = (ngram_count + 1) / (context_count + vocab_size)
                        log_prob_sum += math.log(prob)
                
                return log_prob_sum / len(n_grams) if len(n_grams) > 0 else -float('inf')
        
        # Tokenize into words using NLTK
        tokens = word_tokenize(text.lower())
        
        # Need at least n tokens to create n-grams
        if len(tokens) < n:
            return -10.0 * (n - len(tokens))
        
        # Create n-grams and (n-1)-grams for probability calculation
        n_grams = list(ngrams(tokens, n))
        context_grams = list(ngrams(tokens, n-1)) if n > 1 else None
        
        # Count frequencies
        ngram_freq = FreqDist(n_grams)
        context_freq = FreqDist(context_grams) if context_grams else None
        
        # Calculate log probability with Laplace smoothing
        vocab_size = len(set(tokens))
        log_prob_sum = 0.0
        
        if n == 1:
            # Unigram model
            total_tokens = len(tokens)
            for gram in n_grams:
                count = ngram_freq[gram]
                # Laplace smoothing: (count + 1) / (total + vocab_size)
                prob = (count + 1) / (total_tokens + vocab_size)
                log_prob_sum += math.log(prob)
        else:
            # N-gram model with context
            for i, gram in enumerate(n_grams):
                # Get context (first n-1 words of the n-gram)
                context = context_grams[i] if context_grams else None
                
                # Count of full n-gram
                ngram_count = ngram_freq[gram]
                # Count of context
                context_count = context_freq[context] if context else len(tokens)
                
                # Laplace smoothing: (ngram_count + 1) / (context_count + vocab_size)
                prob = (ngram_count + 1) / (context_count + vocab_size)
                log_prob_sum += math.log(prob)
        
        # Average log probability
        avg_log_prob = log_prob_sum / len(n_grams) if len(n_grams) > 0 else -float('inf')
        
        return avg_log_prob
    
    except Exception as e:
        # If tokenization or n-gram creation fails, return a very low score
        print(f"Warning: N-gram scoring failed for text: {text[:50]}... Error: {e}")
        return -float('inf')

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None, alg_temp=0.0, 
             output_history=False, save_logits=False, question_id=None, **kwargs):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        save_logits: If True, save logits at 1, 2, 4, 8 steps ahead (only works with block_length=1).
        question_id: Question ID for tracking (used when save_logits=True).
    '''
    if save_logits:
        assert block_length == 1, f"save_logits only works with block_length=1 (autoregressive mode), got block_length={block_length}"
    
    history = [] if output_history else None
    saved_logits_data = [] if save_logits else None
    input_length = prompt.shape[1]
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    timestep = 0  # Track global timestep for save_logits
    
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            
            # Save logits at different prediction horizons (before updating x)
            if save_logits:
                # Find the last unmasked position in the generation region
                gen_region_start = prompt.shape[1]
                gen_region_end = prompt.shape[1] + gen_length
                
                # Get mask status for generation region
                gen_mask = (x[:, gen_region_start:gen_region_end] == mask_id)  # (batch, gen_length)
                
                # For each batch (though typically batch=1)
                for b in range(x.shape[0]):
                    # Find last unmasked position in generation region (0-indexed within generation)
                    unmasked_positions = torch.where(~gen_mask[b])[0]
                    
                    if len(unmasked_positions) > 0:
                        last_unmasked_pos = unmasked_positions[-1].item()  # position relative to gen_region_start
                    else:
                        last_unmasked_pos = -1  # No tokens unmasked yet
                    
                    # Save logits for positions 1, 2, 4, 8 steps ahead
                    for horizon in [1, 2, 4, 8]:
                        target_pos = last_unmasked_pos + horizon  # position relative to gen_region_start
                        absolute_pos = gen_region_start + target_pos  # absolute position in x
                        
                        # Check if target position exists and is still masked
                        if target_pos < gen_length and absolute_pos < gen_region_end:
                            if gen_mask[b, target_pos]:  # position is still masked
                                # Save the logits for this position
                                # logits shape: (batch, seq_len, vocab_size)
                                position_logits = logits[b, absolute_pos, :].cpu().clone()
                                
                                # Save metadata and logits separately
                                # Metadata goes in JSON, logits saved separately in safetensors
                                saved_logits_data.append({
                                    'question_id': question_id,
                                    'timestep': timestep,
                                    'horizon': horizon,
                                    'absolute_position': absolute_pos,
                                    'relative_position': target_pos,  # position within generation region
                                    'last_unmasked_position': last_unmasked_pos,
                                    'logits': position_logits,  # Keep as tensor for safetensors
                                })
            
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            
            timestep += 1
            i += 1
            if history is not None:
                history.append(x[:, input_length:].cpu().clone())
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    
    if save_logits:
        return x, nfe, history, saved_logits_data
    else:
        return x, nfe, history



@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1


    return x, nfe


@ torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0  
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            # cache position is the position between current_block_start and current_block_end
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], None, factor)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            i += 1

    return x, nfe


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

def get_top_m_tokens_per_position(logits, temperature, remasking, mask_index, x, num_transfer_tokens, pool_size=1, threshold=None):
    """
    Get top M tokens per position instead of just the top 1.

    Args:
        logits: Model output logits
        temperature: Temperature for Gumbel noise
        remasking: Remasking strategy ('low_confidence' or 'random')
        mask_index: Boolean mask indicating which positions are masked
        x: Current token sequence
        num_transfer_tokens: Number of tokens to transfer at this step
        pool_size: M - number of top tokens to consider per position
        threshold: Optional confidence threshold

    Returns:
        top_m_tokens: Tensor of shape (batch, seq_len, pool_size) with top M tokens per position
        top_m_confidences: Tensor of shape (batch, seq_len, pool_size) with corresponding confidences
        transfer_index: Boolean mask indicating which positions should be transferred
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

    # Get top M tokens per position
    top_m_values, top_m_tokens = torch.topk(logits_with_noise, k=pool_size, dim=-1)  # b, l, M

    # Calculate confidences
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)  # b, l, vocab
        # Get probabilities for top M tokens
        top_m_confidences = torch.gather(p, dim=-1, index=top_m_tokens)  # b, l, M
    elif remasking == 'random':
        top_m_confidences = torch.rand_like(top_m_tokens, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    # For positions that are not masked, we should keep the original token
    # We'll handle this by setting a flag
    transfer_index = mask_index.clone()

    # Determine which positions to transfer based on confidence and num_transfer_tokens
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    # Get the top token's confidence for each position (for determining transfer)
    top1_confidence = top_m_confidences[:, :, 0]  # b, l
    confidence = torch.where(mask_index, top1_confidence, -np.inf)

    # Select positions to transfer based on confidence
    final_transfer_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        final_transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    final_transfer_index[j, select_index[k]] = False

    return top_m_tokens, top_m_confidences, final_transfer_index


@torch.no_grad()
def generate_with_compatibility(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    factor=None,
    pool_size=2,
    tokenizer=None,          # DLM tokenizer (e.g., LLaDA)
    beam_size=32,
    max_joint=32,
    score_function='grammar',
    ngram_size=3,
    output_history=False,
    **kwargs,
):
    """
    Parallel decoding with compatibility via BEAM search using configurable scoring.
    
    Args:
        score_function: Scoring method - 'grammar', 'perplexity', or 'ngram'.
                       - 'grammar': beam score = -grammar_error_count(decoded_text)
                       - 'perplexity': beam score = -perplexity(decoded_text) 
                       - 'ngram': beam score = ngram_score(decoded_text)
        ngram_size: N-gram size for ngram scoring (default: 3 for trigrams).

    Complexity per step: O(M * K * beam_size) scoring evaluations, where
    M = #positions decided jointly this step (<= max_joint), K = pool_size.
    """
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    assert tokenizer is not None, "tokenizer must be provided when pool_size > 1"

    history = [] if output_history else None
    input_length = prompt.shape[1]

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    nfe = 0

    def _beam_pick_by_score(
        x_row: torch.Tensor,
        transfer_positions: torch.Tensor,
        top_m_tokens_row: torch.Tensor,   # (seq, K)
        dlm_tokenizer,
        left_start: int,
        right_end: int,
        beam_size: int,
        score_func: str,
        ngram_n: int,
        device,
    ):
        """
        Return list of (pos, tok_id) chosen via left->right beam.
        Score calculated based on score_func:
        - 'grammar': -grammar_error_count(decoded)
        - 'perplexity': -perplexity(decoded)
        - 'ngram': ngram_score(decoded)
        """
        positions = transfer_positions.tolist()
        positions.sort()

        # beams: list of (assignments) where assignments is list[(pos, tok_id)]
        beams = [[]]

        # simple memo to avoid re-scoring identical partials
        cache = {}  # key: tuple((pos,tok_id),...), val: score

        def score_assign(assign_pairs):
            key = tuple(assign_pairs)
            if key in cache:
                return cache[key]
            temp = x_row.clone()
            for p, t in assign_pairs:
                temp[p] = t
            decoded = dlm_tokenizer.decode(temp[left_start:right_end], skip_special_tokens=True)
            
            try:
                if score_func == 'grammar':
                    err = grammar_error_count(decoded)
                    sc = -float(err)
                elif score_func == 'perplexity':
                    perplexity = calculate_perplexity(decoded, device=device)
                    sc = -perplexity
                elif score_func == 'ngram':
                    sc = calculate_ngram_score(decoded, n=ngram_n)
                else:
                    raise ValueError(f"Unknown score_function: {score_func}. Must be 'grammar', 'perplexity', or 'ngram'")
            except Exception as e:
                # be robust if scorer hiccups
                if score_func in ['grammar', 'perplexity']:
                    sc = -float('inf')  # worst possible score for error-based metrics
                else:
                    sc = -float('inf')  # worst possible score for ngram
            
            cache[key] = sc
            return sc

        for pos in positions:
            expansions = []
            cand_tok_ids = top_m_tokens_row[pos].tolist()

            for chosen in beams:
                for tok_id in cand_tok_ids:
                    new_assn = chosen + [(pos, tok_id)]
                    expansions.append(new_assn)

            # rank expansions by absolute grammar score at this depth
            expansions.sort(key=score_assign, reverse=True)
            beams = expansions[:beam_size]

        best = max(beams, key=score_assign) if beams else []
        return best

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end   = block_start + block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        M = int(block_mask_index.sum().item())
        if M == 0:
            continue

        local_steps = max(1, min(steps, M))
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, local_steps)

        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, block_end:] = 0

            if pool_size == 1:
                # baseline path
                if factor is None:
                    x0, transfer_index = get_transfer_index(
                        logits, temperature, remasking, mask_index, x,
                        num_transfer_tokens[:, min(i, local_steps - 1)] if threshold is None else None,
                        threshold
                    )
                else:
                    x0, transfer_index = get_transfer_index_dynamic(
                        logits, temperature, remasking, mask_index, x, None, factor
                    )
                x[transfer_index] = x0[transfer_index]

            else:
                # compatibility path: top-M per position, then grammar-only beam
                if factor is None:
                    top_m_tokens, top_m_confidences, transfer_index = get_top_m_tokens_per_position(
                        logits, temperature, remasking, mask_index, x,
                        num_transfer_tokens[:, min(i, local_steps - 1)] if threshold is None else None,
                        pool_size, threshold
                    )
                else:
                    x0, transfer_index = get_transfer_index_dynamic(
                        logits, temperature, remasking, mask_index, x, None, factor
                    )
                    top_m_tokens, top_m_confidences, _ = get_top_m_tokens_per_position(
                        logits, temperature, remasking, mask_index, x,
                        transfer_index.sum(dim=1, keepdim=True), pool_size, None
                    )

                B = x.shape[0]
                for b in range(B):
                    pos_all = torch.where(transfer_index[b])[0]

                    if pos_all.numel() == 0:
                        # force progress: pick most confident masked position in block
                        top1 = torch.max(top_m_confidences[b], dim=-1).values  # (seq,)
                        conf = torch.full_like(top1, -float("inf"))
                        conf[block_start:block_end] = top1[block_start:block_end]
                        conf[~mask_index[b]] = -float("inf")
                        pos_all = torch.argmax(conf).unsqueeze(0)

                    pos_all = pos_all.sort().values
                    if pos_all.numel() > max_joint:
                        pos_all = pos_all[:max_joint]

                    chosen = _beam_pick_by_score(
                        x_row=x[b],
                        transfer_positions=pos_all,
                        top_m_tokens_row=top_m_tokens[b],   # (seq, K)
                        dlm_tokenizer=tokenizer,
                        left_start=input_length,            # include prior generated text
                        right_end=block_end,
                        beam_size=beam_size,
                        score_func=score_function,
                        ngram_n=ngram_size,
                        device=model.device,
                    )

                    # apply; safety fallback if empty
                    if not chosen:
                        for pos in pos_all.tolist():
                            x[b, pos] = top_m_tokens[b, pos, 0]
                    else:
                        for pos, tok_id in chosen:
                            x[b, pos] = tok_id

            i += 1
            if history is not None:
                history.append(x[:, input_length:].detach().cpu().clone())
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break

    return x, nfe, history

@ torch.no_grad()
def generate_with_old_compatibility(model, prompt, steps=128, gen_length=128, block_length=128,
                                 temperature=0., remasking='low_confidence', mask_id=126336,
                                 threshold=None, factor=None, pool_size=2, tokenizer=None, 
                                 score_function='perplexity', ngram_size=3, output_history=False, **kwargs):
    '''
    Generate with compatibility checking using grammar, perplexity, or n-gram scores.

    Args:
        model: Mask predictor.
        tokenizer: Tokenizer for decoding tokens to text.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        pool_size: M - number of top tokens to consider per position for compatibility checking.
        threshold: Optional confidence threshold.
        factor: Optional factor for dynamic transfer.
        score_function: Scoring method - 'grammar', 'perplexity', or 'ngram'.
        ngram_size: N-gram size for ngram scoring (default: 3 for trigrams).
        output_history: Whether to output generation history.

    Returns:
        x: Generated sequence
        nfe: Number of function evaluations
        history: Generation history (if output_history=True)
    '''
    # Assert tokenizer is provided when compatibility checking is enabled
    if pool_size > 1:
        assert tokenizer is not None, "tokenizer must be provided when pool_size > 1"

    history = [] if output_history else None
    input_length = prompt.shape[1]

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0

            # If pool_size is 1, use the original method (no compatibility checking)
            if pool_size == 1:
                if factor is None:
                    x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x,
                                                           num_transfer_tokens[:, i] if threshold is None else None, threshold)
                else:
                    x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
                x[transfer_index] = x0[transfer_index]
            else:
                # Use compatibility-based selection
                if factor is None:
                    top_m_tokens, top_m_confidences, transfer_index = get_top_m_tokens_per_position(
                        logits, temperature, remasking, mask_index, x,
                        num_transfer_tokens[:, i] if threshold is None else None, pool_size, threshold
                    )
                else:
                    # For dynamic factor, we first need to determine how many tokens to transfer
                    # We'll use the standard approach but adapt it
                    x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
                    # Now get top M tokens for the positions that will be transferred
                    top_m_tokens, top_m_confidences, _ = get_top_m_tokens_per_position(
                        logits, temperature, remasking, mask_index, x,
                        transfer_index.sum(dim=1, keepdim=True), pool_size, None
                    )

                # Get positions that will be transferred
                batch_size = x.shape[0]
                for b in range(batch_size):
                    transfer_positions = torch.where(transfer_index[b])[0]
                    num_positions = len(transfer_positions)

                    if num_positions == 0:
                        continue

                    # Limit combinatorial explosion: evaluate first max_combinations
                    max_combinations = 100  # Limit to avoid memory issues
                    total_combinations = pool_size ** num_positions

                    # Generate combinations (all if possible, otherwise first max_combinations)
                    candidates = []
                    combo_generator = itertools.product(range(pool_size), repeat=num_positions)

                    for combo_idx, combo_indices in enumerate(combo_generator):
                        if combo_idx >= max_combinations:
                            break

                        candidate = x[b].clone()
                        for pos_idx, token_idx in enumerate(combo_indices):
                            pos = transfer_positions[pos_idx]
                            candidate[pos] = top_m_tokens[b, pos, token_idx]
                        candidates.append(candidate)

                    # Score each candidate using the specified scoring function
                    best_score = -float('inf')
                    best_candidate = None

                    # Check if text is too short for meaningful scoring
                    block_start = prompt.shape[1] + num_block * block_length
                    block_end = prompt.shape[1] + (num_block + 1) * block_length
                    num_generated_tokens = block_end - input_length

                    # If only 1 token or very short, just pick the first candidate
                    if num_generated_tokens <= 1:
                        best_candidate = candidates[0]
                    else:
                        # Score each candidate using the specified scoring function
                        for cand_idx, candidate in enumerate(candidates):
                            # Decode the entire sequence up to current point
                            decoded_text = tokenizer.decode(candidate[input_length:block_end], skip_special_tokens=True)

                            # Calculate score based on the specified scoring function
                            if score_function == 'perplexity':
                                # Calculate perplexity using GPT-2
                                # Lower perplexity = better coherence/compatibility
                                # Use negative perplexity as score (so lower perplexity = higher score)
                                perplexity = calculate_perplexity(decoded_text, device=model.device)
                                score = -perplexity
                            elif score_function == 'grammar':
                                # Calculate grammar score using error count
                                # Negative error count: 0 errors = score 0 (best), more errors = more negative
                                error_count = grammar_error_count(decoded_text)
                                score = -error_count
                            elif score_function == 'ngram':
                                # Calculate n-gram compatibility score
                                # Higher n-gram score = better compatibility (already returns higher = better)
                                score = calculate_ngram_score(decoded_text, n=ngram_size)
                            else:
                                raise ValueError(f"Unknown score_function: {score_function}. Must be 'grammar', 'perplexity', or 'ngram'")

                            if score > best_score:
                                best_score = score
                                best_candidate = candidate

                    # Update x with the best candidate
                    if best_candidate is not None:
                        x[b] = best_candidate


            i += 1
            if history is not None:
                history.append(x[:, input_length:].cpu().clone())
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break

    return x, nfe, history

def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
