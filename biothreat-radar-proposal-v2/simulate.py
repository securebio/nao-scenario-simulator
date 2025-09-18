#!/usr/bin/env python3
"""
This is a simplified port of the HTML+JS simulator to Python, to support the
NAO's proposal for how Biothreat Radar could be implemented.

The key improvement over the main simulator is that it supports combining
multiple systems into a single overall efficacy estimate.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple

# Load the simulator's RAi(1%) distributions.
with open(os.path.dirname(__file__) + "/../html/ww-rai1pct.js") as inf:
    WW_RAI1PCT = json.loads(
        # Convert from JS to JSON
        inf.read().removeprefix("const ww_rai1pct =").removesuffix(";"))

simulation_params = dict(
    doubling_time=3.0,
    cv_doubling_time=0.1,
    shedding_values=[
        "MU-11320", # NWSS
        [
            # Swabs
            5e-6, 5e-6, 6e-6, 7e-6, 1e-5, 1e-5, 2e-5, 3e-5, 3e-5, 3e-5, 4e-5,
            3e-4, 3e-4, 3e-4, 3e-4, 5e-4, 6e-4, 1e-3, 4e-3, 9e-3, 1e-2, 1e-2,
            2e-2, 3e-2, 4e-2, 5e-2, 5e-2, 6e-2, 2e-1, 2e-1, 3e-1, 3e-1, 4e-1,
            6e-1, 6e-1, 7e-1, 2e-7, 9e-7, 2e-5, 1e-5, 1e-5, 7e-5, 5e-5, 1e-2,
            6e-6, 2e-5, 9e-5, 6e-4, 3e-4, 4e-6, 2e-3, 3e-2, 6e-5, 3e-4, 8e-2,
            2e-4, 2e-4, 2e-4, 1e-6, 3e-5, 2e-4, 1e-5, 3e-5, 1e-3,
        ],
        [1.4e-6], # Triturators
        [1.4e-6], # Inidividual Planes
    ],
    sample_populations = [
        500000, # NWSS
        5200,   # Swabs
        97500,  # Triturators
        4500,   # Individual Planes
    ],
    # How deeply do we sequence samples from each source, on a daily basis?
    sample_depths = [
        24e9,   # NWSS
        2e9,    # Swabs
        188e9,  # Triturators
        12e9,   # Individual Planes
    ],
    sigma_shedding_values=0.05,
    shedding_duration=5.0,
    sigma_shedding_duration=0.05,
    processing_delays=[
        2.69, # NWSS
        2.19, # Swabs
        2.65, # Triturators
        2.40, # Individual Planes
    ],
    min_sample_observations=2,
    min_read_observations=2,
    fraction_useful_reads=0.50,
    simulations=100000,
)

# Beyond a 30% infection rate an exponential model gets very inaccurate.
MAX_SUPPORTED_CUMULATIVE_INCIDENCE = 0.3

class BiosurveillanceSimulator:
    def __init__(self, params=None):
        self.rng = np.random.default_rng()
        self.params = params if params is not None else simulation_params

    def inverse_transform_sample(self, cdf: Dict[str, float]) -> float:
        """Sample from a discrete distribution using inverse transform"""
        total_weight = sum(cdf.values())
        target = self.rng.random() * total_weight
        cumsum = 0
        for k, v in cdf.items():
            cumsum += v
            if cumsum >= target:
                return float(k)
        return float(list(cdf.keys())[-1])

    def daily_to_weekly_incidence(self, daily_incidence: float,
                                  growth_factor: float) -> float:
        """Convert daily incidence to weekly incidence accounting for growth"""
        weekly = 0
        effective = daily_incidence
        for _ in range(7):
            weekly += effective
            effective /= growth_factor
        return weekly

    def individual_probability_sick(self, daily_incidence: float,
                                    detectable_days: float,
                                    growth_factor: float) -> float:
        """Calculate probability an individual is currently detectably sick"""
        prob = 0
        effective = daily_incidence
        for _ in range(int(detectable_days)):
            prob += effective
            effective /= growth_factor
        return prob

    def simulate_ra_sick(self, sample_sick: int,
                         ra_sicks: List[float]) -> float:
        """Simulate relative abundance from sick individuals"""
        if sample_sick == 0:
            return 0
        elif len(ra_sicks) == 1:
            return ra_sicks[0]
        elif sample_sick > len(ra_sicks) * 3:
            return np.mean(ra_sicks)
        else:
            return np.mean(self.rng.choice(ra_sicks, size=sample_sick))

    def get_noisy_value(self, mean: float, cv: float) -> float:
        """Add normal noise based on coefficient of variation"""
        if cv < 1e-6:
            return mean
        return self.rng.normal(mean, cv * mean)

    def get_lognormal_value(self, geom_mean: float, sigma: float) -> float:
        """Add lognormal noise"""
        if sigma < 1e-6:
            return geom_mean
        return self.rng.lognormal(np.log(geom_mean), sigma)

    def simulate_one(self) -> float:
        """Run one simulation and return cumulative incidence at detection"""
        day = 0
        population = 1e10

        # Growth parameters with noise
        doubling_time = self.get_noisy_value(self.params['doubling_time'],
                                             self.params['cv_doubling_time'])
        r = np.log(2) / doubling_time
        growth_factor = np.exp(r)
        cumulative_incidence = 1 / population

        # Shedding parameters with noise
        detectable_days = self.get_lognormal_value(
            self.params['shedding_duration'],
            self.params['sigma_shedding_duration']
        )

        ra_sickss = []
        rai1pcts = []
        for shedding_values in self.params['shedding_values']:
            if type(shedding_values) == type(""):
                ra_sickss.append(None)
                rai1pcts.append(self.inverse_transform_sample(
                    WW_RAI1PCT[shedding_values]
                ))
            else:
                # Apply correlated noise to all values
                if self.params['sigma_shedding_values'] > 1e-6:
                    bias = self.rng.lognormal(
                        0, self.params['sigma_shedding_values'])
                    ra_sickss.append([v * bias
                                      for v in shedding_values
                                      if v is not None])
                else:
                    ra_sickss.append(
                        [v for v in shedding_values if v is not None])

                rai1pcts.append(None)

        # Detection tracking
        sample_observations = 0
        read_observations = 0

        # Initialize sites
        site_infos = []
        for _ in ra_sickss:
            site_infos.append({
                'sample_sick': 0,
                'sample_total': 0,
                'sample_weekly_incidences': [],
            })

        # Processing delay factor
        processing_delay_factors = [
            growth_factor ** processing_delay
            for processing_delay in self.params['processing_delays']]

        # Main simulation loop
        while True:
            day += 1
            cumulative_incidence *= growth_factor

            for (site,
                 sample_population,
                 ra_sicks,
                 rai1pct,
                 sample_depth,
                 processing_delay_factor,
                 ) in zip(
                     site_infos,
                     self.params['sample_populations'],
                     ra_sickss,
                     rai1pcts,
                     self.params['sample_depths'],
                     processing_delay_factors,
                 ):

                # Sampling
                if True:
                    daily_incidence = cumulative_incidence * r
                    prob_sick = self.individual_probability_sick(
                        daily_incidence, detectable_days, growth_factor
                    )
                    site['sample_sick'] += self.rng.poisson(
                        sample_population * prob_sick
                    )
                    site['sample_total'] += sample_population
                    site['sample_weekly_incidences'].append(
                        self.daily_to_weekly_incidence(
                            daily_incidence, growth_factor)
                    )

                # Sequencing
                if True:
                    relative_abundance = 0
                    if ra_sicks is not None:
                        relative_abundance = (
                            site['sample_sick'] / site['sample_total'] *
                            self.simulate_ra_sick(site['sample_sick'], ra_sicks)
                        )
                    elif rai1pct is not None:
                        avg_weekly = np.mean(site['sample_weekly_incidences'])
                        relative_abundance = avg_weekly / 0.01 * rai1pct

                    # Reset site accumulation
                    site['sample_sick'] = 0
                    site['sample_total'] = 0
                    site['sample_weekly_incidences'] = []

                    # Check for detection
                    if relative_abundance > 0:
                        prob_useful = self.params[
                            "fraction_useful_reads"] * relative_abundance
                        this_sample_obs = self.rng.poisson(
                            sample_depth * prob_useful
                        )

                        if this_sample_obs > 0:
                            sample_observations += 1
                            read_observations += this_sample_obs

                        if (sample_observations >=
                            self.params['min_sample_observations'] and
                            read_observations >=
                            self.params['min_read_observations']):
                            return cumulative_incidence * processing_delay_factor

            # Check for simulation termination
            if cumulative_incidence > MAX_SUPPORTED_CUMULATIVE_INCIDENCE or \
               day > 365 * 10:
                return 1.0

    def run_simulations(self) -> List[float]:
        """Run multiple simulations and return outcomes"""
        n = self.params['simulations']

        outcomes = []
        for i in range(n):
            if i % 10000 == 0:
                print(f"Simulation {i}/{n}")
            outcomes.append(self.simulate_one())

        return sorted(outcomes)

    def analyze_outcomes(self, outcomes: List[float]) -> Dict:
        """Analyze simulation outcomes and return statistics"""
        percentiles = [25, 50, 75, 90, 95, 99]
        results = {
            'percentiles': {},
            'mean': np.mean(outcomes),
            'std': np.std(outcomes),
            'detection_rate': sum(
                1 for o in outcomes
                if o < MAX_SUPPORTED_CUMULATIVE_INCIDENCE) / len(outcomes)
        }

        for p in percentiles:
            results['percentiles'][p] = np.percentile(outcomes, p)

        return results

def main():
    simulator = BiosurveillanceSimulator()
    results = simulator.analyze_outcomes(simulator.run_simulations())

    print("\nResults:")
    print("Cumulative incidence at detection (percentiles):")
    for p, v in results['percentiles'].items():
        if v < MAX_SUPPORTED_CUMULATIVE_INCIDENCE:
            print(f"  {p}th: {v:.5%}")
        else:
            print(f"  {p}th: Not detected")

if __name__ == "__main__":
    main()
