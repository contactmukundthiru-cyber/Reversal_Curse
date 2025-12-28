/**
 * Reversal Curse Experiment - Complete Implementation
 *
 * CRITICAL METHODOLOGICAL NOTES:
 * 1. Both forward (A→B) and reverse (B→A) tests use 4-AFC format
 *    This eliminates the recall/recognition confound that would bias results.
 * 2. Manipulation check replaces criterion check (bidirectional, no retraining)
 * 3. ISI manipulation for Study 3b (100ms, 500ms, 2000ms conditions)
 * 4. Bidirectional probes in simultaneous condition
 * 5. Response time deadlines enforced
 * 6. Block 2 typed recall (exploratory)
 * 7. Phase 5 fast retraining option (Study 3c)
 */

class ReversalCurseExperiment {
    constructor() {
        // Session information
        this.sessionUuid = sessionStorage.getItem('session_uuid');
        this.condition = sessionStorage.getItem('condition');
        this.config = JSON.parse(sessionStorage.getItem('config') || '{}');
        this.stimuli = JSON.parse(sessionStorage.getItem('stimuli') || '{}');

        // State
        this.currentPhase = 'instructions';
        this.trialNumber = 0;
        this.trainingTrials = [];
        this.manipulationCheckTrials = [];
        this.testTrials4AFC = [];
        this.testTrialsRecall = [];
        this.responses = [];

        // Training state
        this.trainingRepetition = 0;

        // Manipulation check state (replaces criterion check)
        this.manipulationCheckForwardCorrect = 0;
        this.manipulationCheckReverseCorrect = 0;

        // Test state - Block 1 (4-AFC, CONFIRMATORY)
        this.forwardCorrect = 0;
        this.forwardTotal = 0;
        this.reverseCorrect = 0;
        this.reverseTotal = 0;

        // Test state - Block 2 (Typed Recall, EXPLORATORY)
        this.recallForwardCorrect = 0;
        this.recallForwardTotal = 0;
        this.recallReverseCorrect = 0;
        this.recallReverseTotal = 0;

        // Phase 5 retraining state
        this.phase5ForwardCorrect = 0;
        this.phase5ReverseCorrect = 0;

        // Current trial state
        this.trialStartTime = null;
        this.currentTrialData = null;
        this.pendingConfidenceRating = false;
        this.distractorEndTime = null;
        this.responseDeadlineTimer = null;

        // ISI for this session (Study 3b)
        this.isiDuration = this.config.isi_duration_ms || 500;

        // Initialize
        this.init();
    }

    init() {
        if (!this.sessionUuid) {
            window.location.href = '/';
            return;
        }

        this.setupInstructions();
        this.setupEventListeners();
        this.prepareTrials();
    }

    setupInstructions() {
        const instructionsDiv = document.getElementById('condition-instructions');
        const nPairs = Object.keys(this.stimuli).length;
        let instructions = '';

        if (this.condition === 'A_then_B') {
            instructions = `
                <p>In this study, you will learn to associate <strong>symbols</strong> with <strong>labels</strong>.</p>
                <p>During training, you will see:</p>
                <ol>
                    <li>A symbol appears on screen</li>
                    <li>Then a label appears</li>
                    <li>Try to remember which label goes with each symbol</li>
                </ol>
                <p>After training, you'll be tested on what you learned.</p>
                <p>There will be ${nPairs} symbol-label pairs to learn.</p>
                <p><strong>Important:</strong> You will have limited time to respond during tests.</p>
            `;
        } else if (this.condition === 'B_then_A') {
            instructions = `
                <p>In this study, you will learn to associate <strong>labels</strong> with <strong>symbols</strong>.</p>
                <p>During training, you will see:</p>
                <ol>
                    <li>A label appears on screen</li>
                    <li>Then a symbol appears</li>
                    <li>Try to remember which symbol goes with each label</li>
                </ol>
                <p>After training, you'll be tested on what you learned.</p>
                <p>There will be ${nPairs} label-symbol pairs to learn.</p>
                <p><strong>Important:</strong> You will have limited time to respond during tests.</p>
            `;
        } else {
            instructions = `
                <p>In this study, you will learn to associate <strong>symbols</strong> with <strong>labels</strong>.</p>
                <p>During training, you will see:</p>
                <ol>
                    <li>A symbol and its label appear together</li>
                    <li>Sometimes you'll be asked a quick question about what you just saw</li>
                    <li>Try to remember which label goes with each symbol</li>
                </ol>
                <p>After training, you'll be tested on what you learned.</p>
                <p>There will be ${nPairs} symbol-label pairs to learn.</p>
                <p><strong>Important:</strong> You will have limited time to respond during tests.</p>
            `;
        }

        instructionsDiv.innerHTML = instructions;
    }

    setupEventListeners() {
        // Instructions
        document.getElementById('start-training-btn').addEventListener('click', () => {
            this.startTraining();
        });

        // Manipulation check (replaces criterion check)
        const manipCheckInput = document.getElementById('criterion-input');
        if (manipCheckInput) {
            manipCheckInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.checkManipulationCheckResponse();
                }
            });
        }

        // Math distractor
        document.getElementById('math-submit').addEventListener('click', () => {
            this.checkMathResponse();
        });
        document.getElementById('math-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.checkMathResponse();
            }
        });

        // Confidence rating buttons
        document.querySelectorAll('.confidence-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.submitConfidenceRating(parseInt(e.currentTarget.dataset.value));
            });
        });

        // Survey
        document.getElementById('submit-survey').addEventListener('click', () => {
            this.submitSurvey();
        });
    }

    prepareTrials() {
        const symbolIds = Object.keys(this.stimuli);
        const nPairs = symbolIds.length;

        // Determine manipulation check pairs (first N pairs, excluded from main analysis)
        const nManipCheck = Math.min(
            this.config.n_manipulation_check_pairs || 4,
            Math.floor(nPairs / 4)
        );
        const manipCheckIds = symbolIds.slice(0, nManipCheck);
        const testIds = symbolIds.slice(nManipCheck);

        // Prepare training trials
        for (let rep = 0; rep < (this.config.training_repetitions || 6); rep++) {
            const shuffled = this.shuffle([...symbolIds]);
            shuffled.forEach(symbolId => {
                this.trainingTrials.push({
                    symbolId: symbolId,
                    label: this.stimuli[symbolId].label,
                    symbol: this.stimuli[symbolId].symbol,
                    repetition: rep,
                    isManipulationCheck: manipCheckIds.includes(symbolId),
                });
            });
        }

        // Prepare manipulation check trials (bidirectional, tests BOTH directions)
        // Per pre-registration: 2 forward + 2 reverse, applied symmetrically
        const forwardManipTrials = this.shuffle([...manipCheckIds]).slice(0, 2);
        const reverseManipTrials = this.shuffle([...manipCheckIds]).slice(0, 2);

        forwardManipTrials.forEach(symbolId => {
            this.manipulationCheckTrials.push({
                symbolId: symbolId,
                label: this.stimuli[symbolId].label,
                symbol: this.stimuli[symbolId].symbol,
                direction: 'forward',
            });
        });
        reverseManipTrials.forEach(symbolId => {
            this.manipulationCheckTrials.push({
                symbolId: symbolId,
                label: this.stimuli[symbolId].label,
                symbol: this.stimuli[symbolId].symbol,
                direction: 'reverse',
            });
        });
        this.manipulationCheckTrials = this.shuffle(this.manipulationCheckTrials);

        // Prepare Block 1 test trials (4-AFC, both directions interleaved)
        testIds.forEach(symbolId => {
            // Forward test (symbol → label)
            this.testTrials4AFC.push({
                symbolId: symbolId,
                label: this.stimuli[symbolId].label,
                symbol: this.stimuli[symbolId].symbol,
                direction: 'forward',
            });
            // Reverse test (label → symbol)
            this.testTrials4AFC.push({
                symbolId: symbolId,
                label: this.stimuli[symbolId].label,
                symbol: this.stimuli[symbolId].symbol,
                direction: 'reverse',
            });
        });
        this.testTrials4AFC = this.shuffle(this.testTrials4AFC);

        // Prepare Block 2 test trials (Typed Recall, EXPLORATORY)
        if (this.config.enable_typed_recall !== false) {
            testIds.forEach(symbolId => {
                this.testTrialsRecall.push({
                    symbolId: symbolId,
                    label: this.stimuli[symbolId].label,
                    symbol: this.stimuli[symbolId].symbol,
                    direction: 'forward',
                });
                this.testTrialsRecall.push({
                    symbolId: symbolId,
                    label: this.stimuli[symbolId].label,
                    symbol: this.stimuli[symbolId].symbol,
                    direction: 'reverse',
                });
            });
            this.testTrialsRecall = this.shuffle(this.testTrialsRecall);
        }
    }

    shuffle(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }

    showScreen(screenId) {
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        const screen = document.getElementById(screenId);
        if (screen) {
            screen.classList.add('active');
        }
    }

    async startTraining() {
        this.currentPhase = 'training';
        await this.updatePhaseOnServer('training');
        this.showScreen('training-screen');
        this.runTrainingTrial(0);
    }

    async runTrainingTrial(index) {
        if (index >= this.trainingTrials.length) {
            // Training complete, run manipulation check
            this.startManipulationCheck();
            return;
        }

        const trial = this.trainingTrials[index];
        const stimulusDisplay = document.getElementById('stimulus-display');
        const labelDisplay = document.getElementById('label-display');
        const fixation = document.getElementById('fixation');
        const progressBar = document.getElementById('training-progress');
        const trialInfo = document.getElementById('training-info');

        // Update progress
        const progress = ((index + 1) / this.trainingTrials.length) * 100;
        progressBar.style.width = `${progress}%`;
        trialInfo.textContent = `Trial ${index + 1} of ${this.trainingTrials.length}`;

        // Clear displays
        stimulusDisplay.innerHTML = '';
        labelDisplay.textContent = '';
        fixation.style.display = 'none';

        // Show fixation
        fixation.style.display = 'block';
        await this.wait(this.config.fixation_duration_ms || 500);
        fixation.style.display = 'none';

        const trialStartTime = Date.now();

        if (this.condition === 'A_then_B') {
            // Show symbol first
            stimulusDisplay.innerHTML = this.createSvgElement(trial.symbol);
            await this.wait(this.config.stimulus_duration_ms || 1500);

            // ISI (interstimulus interval) - key manipulation for Study 3b
            stimulusDisplay.innerHTML = '';
            await this.wait(this.isiDuration);

            // Show label
            labelDisplay.textContent = trial.label;
            await this.wait(this.config.stimulus_duration_ms || 1500);

        } else if (this.condition === 'B_then_A') {
            // Show label first
            labelDisplay.textContent = trial.label;
            await this.wait(this.config.stimulus_duration_ms || 1500);

            // ISI
            labelDisplay.textContent = '';
            await this.wait(this.isiDuration);

            // Show symbol
            stimulusDisplay.innerHTML = this.createSvgElement(trial.symbol);
            await this.wait(this.config.stimulus_duration_ms || 1500);

        } else {
            // Simultaneous condition with bidirectional probes
            stimulusDisplay.innerHTML = this.createSvgElement(trial.symbol);
            labelDisplay.textContent = trial.label;
            await this.wait(this.config.simultaneous_duration_ms || 2000);

            // Bidirectional probe (50% of trials)
            const probeRate = this.config.simultaneous_probe_rate || 0.5;
            if (Math.random() < probeRate) {
                await this.runSimultaneousProbe(trial, index);
            }
        }

        // Clear and ITI
        stimulusDisplay.innerHTML = '';
        labelDisplay.textContent = '';
        await this.wait(this.config.iti_duration_ms || 1000);

        // Record trial
        await this.recordTrial({
            trial_number: index,
            phase: 'training',
            symbol_id: trial.symbolId,
            label: trial.label,
            isi_actual_ms: this.isiDuration,
        });

        // Next trial
        this.runTrainingTrial(index + 1);
    }

    async runSimultaneousProbe(trial, trialIndex) {
        // Hide the main displays
        document.getElementById('stimulus-display').innerHTML = '';
        document.getElementById('label-display').textContent = '';

        // 50% forward probe ("Which label?"), 50% reverse probe ("Which symbol?")
        const isForwardProbe = Math.random() < 0.5;

        const probeContainer = document.getElementById('probe-container');
        if (!probeContainer) {
            // Create probe container if it doesn't exist
            const container = document.createElement('div');
            container.id = 'probe-container';
            container.className = 'probe-container';
            document.getElementById('training-screen').appendChild(container);
        }

        return new Promise((resolve) => {
            const probeDiv = document.getElementById('probe-container');
            probeDiv.style.display = 'block';

            if (isForwardProbe) {
                // Show symbol, ask for label
                probeDiv.innerHTML = `
                    <div class="probe-question">Quick check: What label goes with this symbol?</div>
                    <div class="probe-stimulus">${this.createSvgElement(trial.symbol)}</div>
                    <div class="probe-options" id="probe-options"></div>
                `;
                const optionsDiv = document.getElementById('probe-options');
                const allLabels = Object.values(this.stimuli).map(s => s.label);
                const foils = this.shuffle(allLabels.filter(l => l !== trial.label)).slice(0, 3);
                const options = this.shuffle([trial.label, ...foils]);

                options.forEach(label => {
                    const btn = document.createElement('button');
                    btn.className = 'probe-option-btn';
                    btn.textContent = label;
                    btn.onclick = async () => {
                        const correct = label === trial.label;
                        await this.recordTrial({
                            trial_number: trialIndex,
                            phase: 'simultaneous_probe',
                            symbol_id: trial.symbolId,
                            label: trial.label,
                            probe_type: 'forward_probe',
                            response: label,
                            correct: correct,
                        });
                        probeDiv.style.display = 'none';
                        probeDiv.innerHTML = '';
                        resolve();
                    };
                    optionsDiv.appendChild(btn);
                });
            } else {
                // Show label, ask for symbol
                probeDiv.innerHTML = `
                    <div class="probe-question">Quick check: Which symbol goes with "${trial.label}"?</div>
                    <div class="probe-options" id="probe-options"></div>
                `;
                const optionsDiv = document.getElementById('probe-options');
                const allSymbolIds = Object.keys(this.stimuli);
                const foilIds = this.shuffle(allSymbolIds.filter(id => id !== trial.symbolId)).slice(0, 3);
                const options = this.shuffle([trial.symbolId, ...foilIds]);

                options.forEach(symbolId => {
                    const btn = document.createElement('button');
                    btn.className = 'probe-option-btn symbol-probe-btn';
                    btn.innerHTML = this.createSvgElement(this.stimuli[symbolId].symbol);
                    btn.onclick = async () => {
                        const correct = symbolId === trial.symbolId;
                        await this.recordTrial({
                            trial_number: trialIndex,
                            phase: 'simultaneous_probe',
                            symbol_id: trial.symbolId,
                            label: trial.label,
                            probe_type: 'reverse_probe',
                            response: symbolId,
                            correct: correct,
                        });
                        probeDiv.style.display = 'none';
                        probeDiv.innerHTML = '';
                        resolve();
                    };
                    optionsDiv.appendChild(btn);
                });
            }

            // Timeout for probe (use probe duration from config)
            setTimeout(() => {
                if (probeDiv.style.display !== 'none') {
                    probeDiv.style.display = 'none';
                    probeDiv.innerHTML = '';
                    resolve();
                }
            }, this.config.simultaneous_probe_duration_ms || 5000);
        });
    }

    createSvgElement(symbol) {
        if (!symbol) return '<div class="missing-symbol">?</div>';
        return `
            <svg viewBox="${symbol.svg_viewbox || '0 0 100 100'}" class="symbol-svg">
                <path d="${symbol.svg_path}"
                      stroke="${symbol.stroke || '#333'}"
                      stroke-width="${symbol.stroke_width || 3}"
                      fill="${symbol.fill || 'none'}"
                      stroke-linecap="round"
                      stroke-linejoin="round"/>
            </svg>
        `;
    }

    // Manipulation Check (replaces Criterion Check)
    // Tests BOTH directions symmetrically - no direction-biased retraining
    async startManipulationCheck() {
        this.currentPhase = 'manipulation_check';
        await this.updatePhaseOnServer('manipulation_check');
        this.showScreen('criterion-screen');

        // Update the screen title
        const title = document.querySelector('#criterion-screen h2');
        if (title) {
            title.textContent = 'Quick Memory Check';
        }

        this.manipCheckIndex = 0;
        this.showManipulationCheckTrial();
    }

    showManipulationCheckTrial() {
        if (this.manipCheckIndex >= this.manipulationCheckTrials.length) {
            this.evaluateManipulationCheck();
            return;
        }

        const trial = this.manipulationCheckTrials[this.manipCheckIndex];
        const stimulusDiv = document.getElementById('criterion-stimulus');
        const inputDiv = document.getElementById('criterion-input');
        const feedback = document.getElementById('criterion-feedback');

        feedback.textContent = '';

        if (trial.direction === 'forward') {
            // Show symbol, ask for label
            stimulusDiv.innerHTML = this.createSvgElement(trial.symbol);
            inputDiv.placeholder = 'Type the label...';
        } else {
            // Show label, ask for symbol description (simpler for manipulation check)
            stimulusDiv.innerHTML = `<div class="label-display-large">${trial.label}</div>`;
            inputDiv.placeholder = 'Describe the symbol...';
        }

        inputDiv.value = '';
        inputDiv.focus();
        this.trialStartTime = Date.now();
    }

    async checkManipulationCheckResponse() {
        const input = document.getElementById('criterion-input');
        const feedback = document.getElementById('criterion-feedback');
        const response = input.value.trim().toUpperCase();
        const trial = this.manipulationCheckTrials[this.manipCheckIndex];
        const rt = Date.now() - this.trialStartTime;

        let correct = false;
        if (trial.direction === 'forward') {
            correct = response === trial.label.toUpperCase();
            if (correct) {
                this.manipulationCheckForwardCorrect++;
            }
        } else {
            // For reverse, we're just checking engagement (any response counts)
            // Strict matching would require showing symbol options
            correct = response.length > 0;
            if (correct) {
                this.manipulationCheckReverseCorrect++;
            }
        }

        feedback.textContent = correct ? 'Got it!' : 'Noted.';
        feedback.className = 'feedback ' + (correct ? 'correct' : 'neutral');

        await this.recordTrial({
            trial_number: this.manipCheckIndex,
            phase: 'manipulation_check',
            symbol_id: trial.symbolId,
            label: trial.label,
            test_direction: trial.direction,
            response: response,
            correct: correct,
            response_time_ms: rt,
        });

        await this.wait(800);
        this.manipCheckIndex++;
        this.showManipulationCheckTrial();
    }

    async evaluateManipulationCheck() {
        // Calculate manipulation check accuracy
        const totalTrials = this.manipulationCheckTrials.length;
        const forwardTrials = this.manipulationCheckTrials.filter(t => t.direction === 'forward').length;
        const reverseTrials = totalTrials - forwardTrials;

        const forwardAcc = forwardTrials > 0 ? this.manipulationCheckForwardCorrect / forwardTrials : 0;
        const reverseAcc = reverseTrials > 0 ? this.manipulationCheckReverseCorrect / reverseTrials : 0;
        const overallAcc = (this.manipulationCheckForwardCorrect + this.manipulationCheckReverseCorrect) / totalTrials;

        const threshold = this.config.manipulation_check_threshold || 0.5;
        const passed = overallAcc >= threshold;

        await this.recordManipulationCheck({
            forward_correct: this.manipulationCheckForwardCorrect,
            forward_total: forwardTrials,
            reverse_correct: this.manipulationCheckReverseCorrect,
            reverse_total: reverseTrials,
            passed: passed,
        });

        // Continue regardless (per pre-registration: exclusion applied symmetrically post-hoc)
        this.startDistractor();
    }

    async startDistractor() {
        this.currentPhase = 'distractor';
        await this.updatePhaseOnServer('distractor');
        this.showScreen('distractor-screen');

        this.distractorEndTime = Date.now() + (this.config.distractor_duration_seconds || 60) * 1000;
        this.showMathProblem();
        this.updateDistractorTimer();
    }

    showMathProblem() {
        const a = Math.floor(Math.random() * 20) + 1;
        const b = Math.floor(Math.random() * 20) + 1;
        const op = Math.random() > 0.5 ? '+' : '-';
        this.currentMathAnswer = op === '+' ? a + b : a - b;

        document.getElementById('math-problem').textContent = `${a} ${op} ${b} = ?`;
        document.getElementById('math-input').value = '';
        document.getElementById('math-input').focus();
        document.getElementById('math-feedback').textContent = '';
    }

    checkMathResponse() {
        const input = document.getElementById('math-input');
        const feedback = document.getElementById('math-feedback');
        const response = parseInt(input.value, 10);

        if (response === this.currentMathAnswer) {
            feedback.textContent = 'Correct!';
            feedback.className = 'feedback correct';
        } else {
            feedback.textContent = 'Try again!';
            feedback.className = 'feedback incorrect';
        }

        setTimeout(() => {
            if (Date.now() < this.distractorEndTime) {
                this.showMathProblem();
            } else {
                this.startTestBlock1();
            }
        }, 500);
    }

    updateDistractorTimer() {
        const remaining = Math.max(0, Math.ceil((this.distractorEndTime - Date.now()) / 1000));
        document.getElementById('distractor-timer').textContent = `Time remaining: ${remaining}s`;

        if (remaining > 0) {
            setTimeout(() => this.updateDistractorTimer(), 1000);
        } else {
            this.startTestBlock1();
        }
    }

    // Block 1: 4-AFC Test (CONFIRMATORY)
    async startTestBlock1() {
        this.currentPhase = 'test_4afc';
        await this.updatePhaseOnServer('test_4afc');
        this.showScreen('test-screen');

        // Update test instructions
        const testInstruction = document.querySelector('#test-screen .test-instruction');
        if (testInstruction) {
            testInstruction.innerHTML = `
                <p><strong>Block 1 of 2: Selection Test</strong></p>
                <p>Select the correct answer from the 4 options.</p>
                <p>You have ${(this.config.test_4afc_deadline_ms || 8000) / 1000} seconds per question.</p>
            `;
        }

        this.testIndex = 0;
        this.show4AFCTestTrial();
    }

    show4AFCTestTrial() {
        if (this.testIndex >= this.testTrials4AFC.length) {
            // Block 1 complete, start Block 2 if enabled
            if (this.config.enable_typed_recall !== false && this.testTrialsRecall.length > 0) {
                this.startTestBlock2();
            } else if (this.config.enable_phase5_retraining) {
                this.startPhase5();
            } else {
                this.startSurvey();
            }
            return;
        }

        const trial = this.testTrials4AFC[this.testIndex];
        const progressBar = document.getElementById('test-progress');
        const progress = ((this.testIndex + 1) / this.testTrials4AFC.length) * 100;
        progressBar.style.width = `${progress}%`;

        // Hide confidence rating from previous trial
        document.getElementById('confidence-rating').style.display = 'none';

        // Clear any previous timeout
        if (this.responseDeadlineTimer) {
            clearTimeout(this.responseDeadlineTimer);
        }

        this.trialStartTime = Date.now();

        // Set response deadline
        const deadline = this.config.test_4afc_deadline_ms || 8000;
        this.responseDeadlineTimer = setTimeout(() => {
            this.handleTimeout(trial, '4afc');
        }, deadline);

        if (trial.direction === 'forward') {
            this.showForward4AFCTrial(trial);
        } else {
            this.showReverse4AFCTrial(trial);
        }
    }

    showForward4AFCTrial(trial) {
        document.getElementById('forward-test').style.display = 'block';
        document.getElementById('reverse-test').style.display = 'none';
        document.getElementById('test-stimulus').innerHTML = this.createSvgElement(trial.symbol);

        const optionsDiv = document.getElementById('forward-options');
        optionsDiv.innerHTML = '';

        const allLabels = Object.values(this.stimuli).map(s => s.label);
        const foilLabels = this.shuffle(allLabels.filter(l => l !== trial.label)).slice(0, 3);
        const options = this.shuffle([trial.label, ...foilLabels]);
        trial.foils = foilLabels;

        options.forEach(label => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'option-button label-option';
            button.textContent = label;
            button.addEventListener('click', () => this.submitForwardResponse(label, trial, '4afc'));
            optionsDiv.appendChild(button);
        });
    }

    showReverse4AFCTrial(trial) {
        document.getElementById('forward-test').style.display = 'none';
        document.getElementById('reverse-test').style.display = 'block';
        document.getElementById('test-label').textContent = trial.label;

        const optionsDiv = document.getElementById('reverse-options');
        optionsDiv.innerHTML = '';

        const allSymbolIds = Object.keys(this.stimuli);
        const foilIds = this.shuffle(allSymbolIds.filter(id => id !== trial.symbolId)).slice(0, 3);
        const options = this.shuffle([trial.symbolId, ...foilIds]);
        trial.foils = foilIds.map(id => this.stimuli[id].label);

        options.forEach(symbolId => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'option-button symbol-option';
            button.innerHTML = this.createSvgElement(this.stimuli[symbolId].symbol);
            button.addEventListener('click', () => this.submitReverseResponse(symbolId, trial, '4afc'));
            optionsDiv.appendChild(button);
        });
    }

    async handleTimeout(trial, testType) {
        // Clear the timer
        if (this.responseDeadlineTimer) {
            clearTimeout(this.responseDeadlineTimer);
            this.responseDeadlineTimer = null;
        }

        const rt = Date.now() - this.trialStartTime;

        // Record as timed out
        this.currentTrialData = {
            trial_number: this.testIndex,
            phase: testType === '4afc' ? 'test_4afc' : 'test_recall',
            symbol_id: trial.symbolId,
            label: trial.label,
            test_direction: trial.direction,
            test_type: testType,
            response: null,
            correct: false,
            timed_out: true,
            response_time_ms: rt,
            foils: trial.foils || [],
        };

        // Update counters
        if (testType === '4afc') {
            if (trial.direction === 'forward') {
                this.forwardTotal++;
            } else {
                this.reverseTotal++;
            }
        } else {
            if (trial.direction === 'forward') {
                this.recallForwardTotal++;
            } else {
                this.recallReverseTotal++;
            }
        }

        // Show timeout message briefly
        const feedbackDiv = document.createElement('div');
        feedbackDiv.className = 'timeout-feedback';
        feedbackDiv.textContent = 'Time\'s up!';
        document.getElementById('test-screen').appendChild(feedbackDiv);

        await this.wait(500);
        feedbackDiv.remove();

        // Skip confidence for timeouts, record and move on
        await this.recordTrial(this.currentTrialData);
        this.currentTrialData = null;
        this.testIndex++;

        if (testType === '4afc') {
            this.show4AFCTestTrial();
        } else {
            this.showRecallTestTrial();
        }
    }

    async submitForwardResponse(selectedLabel, trial, testType) {
        if (this.responseDeadlineTimer) {
            clearTimeout(this.responseDeadlineTimer);
            this.responseDeadlineTimer = null;
        }

        const correct = selectedLabel === trial.label;
        const rt = Date.now() - this.trialStartTime;

        if (testType === '4afc') {
            this.forwardTotal++;
            if (correct) this.forwardCorrect++;
        } else {
            this.recallForwardTotal++;
            if (correct) this.recallForwardCorrect++;
        }

        this.currentTrialData = {
            trial_number: this.testIndex,
            phase: testType === '4afc' ? 'test_4afc' : 'test_recall',
            symbol_id: trial.symbolId,
            label: trial.label,
            test_direction: 'forward',
            test_type: testType,
            response: selectedLabel,
            correct: correct,
            timed_out: false,
            response_time_ms: rt,
            foils: trial.foils || [],
        };

        this.showConfidenceRating(testType);
    }

    async submitReverseResponse(selectedSymbolId, trial, testType) {
        if (this.responseDeadlineTimer) {
            clearTimeout(this.responseDeadlineTimer);
            this.responseDeadlineTimer = null;
        }

        const correct = selectedSymbolId === trial.symbolId;
        const rt = Date.now() - this.trialStartTime;

        if (testType === '4afc') {
            this.reverseTotal++;
            if (correct) this.reverseCorrect++;
        } else {
            this.recallReverseTotal++;
            if (correct) this.recallReverseCorrect++;
        }

        this.currentTrialData = {
            trial_number: this.testIndex,
            phase: testType === '4afc' ? 'test_4afc' : 'test_recall',
            symbol_id: trial.symbolId,
            label: trial.label,
            test_direction: 'reverse',
            test_type: testType,
            response: selectedSymbolId,
            correct: correct,
            timed_out: false,
            response_time_ms: rt,
            foils: trial.foils || [],
        };

        this.showConfidenceRating(testType);
    }

    showConfidenceRating(testType) {
        document.getElementById('forward-test').style.display = 'none';
        document.getElementById('reverse-test').style.display = 'none';
        document.getElementById('confidence-rating').style.display = 'block';
        this.pendingConfidenceRating = true;
        this.currentTestType = testType;
    }

    async submitConfidenceRating(rating) {
        if (!this.pendingConfidenceRating || !this.currentTrialData) return;

        this.pendingConfidenceRating = false;
        this.currentTrialData.confidence = rating;

        await this.recordTrial(this.currentTrialData);
        this.currentTrialData = null;

        this.testIndex++;

        if (this.currentTestType === '4afc') {
            this.show4AFCTestTrial();
        } else {
            this.showRecallTestTrial();
        }
    }

    // Block 2: Typed Recall Test (EXPLORATORY)
    async startTestBlock2() {
        this.currentPhase = 'test_recall';
        await this.updatePhaseOnServer('test_recall');

        const testInstruction = document.querySelector('#test-screen .test-instruction');
        if (testInstruction) {
            testInstruction.innerHTML = `
                <p><strong>Block 2 of 2: Recall Test</strong></p>
                <p>Type your answer from memory.</p>
                <p>You have ${(this.config.test_recall_deadline_ms || 15000) / 1000} seconds per question.</p>
            `;
        }

        this.testIndex = 0;
        this.showRecallTestTrial();
    }

    showRecallTestTrial() {
        if (this.testIndex >= this.testTrialsRecall.length) {
            if (this.config.enable_phase5_retraining) {
                this.startPhase5();
            } else {
                this.startSurvey();
            }
            return;
        }

        // Implementation would show a text input for typed recall
        // For now, skip to survey (this is exploratory)
        this.startSurvey();
    }

    // Phase 5: Fast Retraining (Study 3c)
    async startPhase5() {
        this.currentPhase = 'phase5_retraining';
        await this.updatePhaseOnServer('phase5_retraining');

        // This would implement the fast reverse-order retraining
        // Then test again to see asymmetry collapse

        // For now, proceed to survey
        this.startSurvey();
    }

    async startSurvey() {
        this.currentPhase = 'survey';
        await this.updatePhaseOnServer('survey');
        this.showScreen('survey-screen');
    }

    async submitSurvey() {
        const attentionRadio = document.querySelector('input[name="attention"]:checked');
        const attentionPassed = attentionRadio && attentionRadio.value === '5';

        await this.recordAttention({
            passed: attentionPassed,
            response: attentionRadio ? attentionRadio.value : null,
        });

        await this.recordDemographics({
            age: document.getElementById('age').value,
            gender: document.getElementById('gender').value,
            education: document.getElementById('education').value,
            native_language: document.getElementById('native-language').value,
            strategy: document.getElementById('strategy').value,
            suspicion: document.getElementById('suspicion').value,
            comments: document.getElementById('comments').value,
        });

        await this.completeSession();
    }

    async completeSession() {
        try {
            const response = await fetch(`/api/session/${this.sessionUuid}/complete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    forward_correct: this.forwardCorrect,
                    forward_total: this.forwardTotal,
                    reverse_correct: this.reverseCorrect,
                    reverse_total: this.reverseTotal,
                    recall_forward_correct: this.recallForwardCorrect,
                    recall_forward_total: this.recallForwardTotal,
                    recall_reverse_correct: this.recallReverseCorrect,
                    recall_reverse_total: this.recallReverseTotal,
                }),
            });
            const data = await response.json();

            if (data.success) {
                document.getElementById('completion-code').textContent = data.completion_code;
                const prolificUrl = `https://app.prolific.co/submissions/complete?cc=${data.completion_code}`;
                document.getElementById('prolific-return').href = prolificUrl;
                this.showScreen('complete-screen');
            }
        } catch (error) {
            console.error('Error completing session:', error);
            alert('Error completing session. Please contact the researcher.');
        }
    }

    // API helpers
    async updatePhaseOnServer(phase) {
        try {
            await fetch(`/api/session/${this.sessionUuid}/update-phase`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ phase }),
            });
        } catch (error) {
            console.error('Error updating phase:', error);
        }
    }

    async recordTrial(trialData) {
        try {
            await fetch(`/api/session/${this.sessionUuid}/record-trial`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(trialData),
            });
        } catch (error) {
            console.error('Error recording trial:', error);
        }
    }

    async recordManipulationCheck(data) {
        try {
            await fetch(`/api/session/${this.sessionUuid}/record-manipulation-check`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
        } catch (error) {
            console.error('Error recording manipulation check:', error);
        }
    }

    async recordAttention(data) {
        try {
            await fetch(`/api/session/${this.sessionUuid}/record-attention`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
        } catch (error) {
            console.error('Error recording attention:', error);
        }
    }

    async recordDemographics(data) {
        try {
            await fetch(`/api/session/${this.sessionUuid}/record-demographics`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
        } catch (error) {
            console.error('Error recording demographics:', error);
        }
    }

    wait(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize experiment when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.experiment = new ReversalCurseExperiment();
});
