# R1-C3 / R2-C2: frozen protocol, 2026-10-15

Experimental arm: economic-only **service provider** DDQN with **unchanged capacity-aware HEMS**.
This does not test a fully capacity-blind household controller.

- Frozen reference: final consolidated CADR project (14 Oct), original Seeds 0,1,2.
- New seeds: 0,1,2; 2,500 episodes, validation every 50; model selection ONLY via own validation mean reward; July/August full-month tests.
- SP observation: 43 dimensions, two final capacity-need inputs forcibly zero, all other variables and normalisations unchanged (division by 7 is fixed scaling, not a capacity decision rule).
- Reward: omega*[(p*settled_DR-payment)/S] +(1-omega)*[rho*payment/S-(1-rho)*class_normalised_dissatisfaction]; omega=0.65, rho=0.90, S=32.8526 derived from unchanged config. No tracking and no no-need penalty.
- The physical HEMS **retains** its capacity-based EV shifting amount and destination headroom, TS-NI destination screening, deadlines, device curtailment choices, locked commitments, and verified settlement.
- Training: same seeds/data/hyperparameters/budget as each matched CADR model. Policy weights vary due to learning; no test data during selection.
- Primary outcomes: July and August aggregate post-DR exceedance hours (>7 kW + 1e-6), cumulative excess kWh. Report paired per-seed values and mean +/- sample SD.
- Secondary outcomes: useful DR, curtailment, shifted/unserved energy, newly created and rebound violations, daily peak and PAR, offers in no-need hours, incentive payments, reported SP economic value and EU utility, net wholesale procurement effect where available.
- Compare physical/economic reported metrics rather than raw rewards across arms. No numerical success criterion or post-test hyperparameter tuning.
- Mandatory gates BEFORE real training: 43-state mask test, hand-computed reward equality, identical physics+settlement under matched actions, retained four no-look-ahead tests, original env.py SHA unchanged.
- Optional rule-based benchmark is not part of this preregistration.
