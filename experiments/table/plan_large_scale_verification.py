#!/usr/bin/env python3
"""
Plan for large scale run to verify: loss = g(L/freq) with optimal range 7-12
We need to test many L values for each frequency to build the precise curve
"""
import numpy as np

def generate_verification_plan():
    """we generate a plan to verify the scaling law hypothesis"""
    
    print("="*80)
    print("LARGE SCALE RUN PLAN: Verify loss = g(L/freq) with optimal range 7-12")
    print("="*80)
    
    # we define frequencies to test
    # focus on low frequencies with many L values
    freq_multipliers = [0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
    
    print("\n📊 STRATEGY:")
    print("  For each frequency, test MANY layer values to cover L/freq range 4-20")
    print("  This will give us a smooth curve: loss = g(L/freq)")
    print("  Goal: verify U-shaped curve with minimum in range 7-12")
    
    print("\n🎯 FREQUENCIES TO TEST:")
    for freq in freq_multipliers:
        # we compute L range to cover L/freq from 4 to 20
        # L/freq = 4 means L = 4 * freq
        # L/freq = 20 means L = 20 * freq
        L_min = max(3, int(np.ceil(4 * freq)))  # at least 3, but aim for L/freq >= 4
        L_max = int(np.ceil(20 * freq))  # aim for L/freq <= 20
        
        # we generate many L values to densely cover the range
        if freq <= 0.3:
            # for very low frequencies, we need to test many small L values
            # but also ensure we cover the range
            L_values = list(range(max(3, L_min), min(L_max + 1, 30), 1))
            # we also add specific values to ensure coverage
            for target_ratio in [7, 8, 9, 10, 11, 12]:
                target_L = int(np.round(target_ratio * freq))
                if target_L >= 3 and target_L <= 30 and target_L not in L_values:
                    L_values.append(target_L)
        elif freq <= 1.0:
            L_values = list(range(max(3, L_min), min(L_max + 1, 40), 1))
            # ensure optimal range coverage
            for target_ratio in [7, 8, 9, 10, 11, 12]:
                target_L = int(np.round(target_ratio * freq))
                if target_L >= 3 and target_L <= 40 and target_L not in L_values:
                    L_values.append(target_L)
        else:
            L_values = list(range(max(3, L_min), min(L_max + 1, 60), 2))  # every 2 layers
            # ensure optimal range coverage
            for target_ratio in [7, 8, 9, 10, 11, 12]:
                target_L = int(np.round(target_ratio * freq))
                if target_L >= 3 and target_L <= 60 and target_L not in L_values:
                    L_values.append(target_L)
        
        L_values = sorted(set(L_values))
        
        print(f"\n  freq×{freq:.2f}:")
        print(f"    L range: {L_min} to {L_max}")
        print(f"    L values: {len(L_values)} configs")
        print(f"    L/freq range: {L_min/freq:.1f} to {L_max/freq:.1f}")
        print(f"    Optimal range (7-12) → L: {optimal_L_min} to {optimal_L_max}")
        print(f"    Sample L values: {L_values[:10]}..." if len(L_values) > 10 else f"    L values: {L_values}")
    
    # we compute total
    total_configs = 0
    for freq in freq_multipliers:
        L_min = max(3, int(4 * freq))
        L_max = int(20 * freq) + 1
        if freq <= 0.3:
            num_L = min(L_max - L_min, 25 - L_min)
        elif freq <= 1.0:
            num_L = min(L_max - L_min, 30 - L_min)
        else:
            num_L = (L_max - L_min) // 2 + 1
        total_configs += num_L * 3  # 3 ranks
    
    print(f"\n📈 ESTIMATED TOTAL CONFIGURATIONS: ~{total_configs}")
    print(f"   This will provide dense coverage of L/freq ratios")
    print(f"   Enough to verify the U-shaped curve hypothesis")
    
    return freq_multipliers

def create_config_generator():
    """we create a function to generate configurations"""
    
    configs = []
    freq_multipliers = [0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
    ranks = [10, 15, 25]
    
    for freq in freq_multipliers:
        # we compute L range to cover L/freq from 4 to 20
        L_min = max(3, int(4 * freq))
        L_max = int(20 * freq) + 1
        
        # we generate L values to cover L/freq from 4 to 20
        L_min = max(3, int(np.ceil(4 * freq)))
        L_max = int(np.ceil(20 * freq))
        
        if freq <= 0.3:
            L_values = list(range(max(3, L_min), min(L_max + 1, 30), 1))
            for target_ratio in [7, 8, 9, 10, 11, 12]:
                target_L = int(np.round(target_ratio * freq))
                if target_L >= 3 and target_L <= 30 and target_L not in L_values:
                    L_values.append(target_L)
        elif freq <= 1.0:
            L_values = list(range(max(3, L_min), min(L_max + 1, 40), 1))
            for target_ratio in [7, 8, 9, 10, 11, 12]:
                target_L = int(np.round(target_ratio * freq))
                if target_L >= 3 and target_L <= 40 and target_L not in L_values:
                    L_values.append(target_L)
        else:
            L_values = list(range(max(3, L_min), min(L_max + 1, 60), 2))
            for target_ratio in [7, 8, 9, 10, 11, 12]:
                target_L = int(np.round(target_ratio * freq))
                if target_L >= 3 and target_L <= 60 and target_L not in L_values:
                    L_values.append(target_L)
        
        L_values = sorted(set(L_values))
        
        for rank in ranks:
            for L in L_values:
                # we compute epochs
                num_epochs = min(int(2 * freq * 10000), 200000)
                num_epochs = max(num_epochs, 5000)
                
                config = {
                    'freq_multiplier': freq,
                    'hidden_rank': rank,
                    'num_layers': L,
                    'batch_size': 100,
                    'num_epochs': num_epochs,
                    'L_over_freq': L / freq,
                }
                configs.append(config)
    
    return configs

if __name__ == "__main__":
    freq_multipliers = generate_verification_plan()
    configs = create_config_generator()
    
    print(f"\n✅ Generated {len(configs)} configurations")
    print(f"   This will provide comprehensive coverage to verify the hypothesis")
    
    # we save the plan
    import json
    plan = {
        'frequencies': freq_multipliers,
        'total_configs': len(configs),
        'goal': 'Verify loss = g(L/freq) with optimal range 7-12',
        'strategy': 'Test many L values for each frequency to cover L/freq range 4-20'
    }
    
    with open('experiments/table/large_scale_verification_plan.json', 'w') as f:
        json.dump(plan, f, indent=4)
    
    print(f"\n✓ Plan saved to: experiments/table/large_scale_verification_plan.json")
