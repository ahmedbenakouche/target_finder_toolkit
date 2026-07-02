# Experiment Change Log

## 2026-06-30

- Hardened the full-experiment instruction, calibration, pause, and between-task screens.
- Rationale: fullscreen wait screens could appear stuck if the continue button did not exit the nested event loop or if Q/Esc focus was captured elsewhere. The shared session screen now exits its active wait loop directly and has a global Q/Esc fallback while visible.

- Added text-based Q handling to the realistic screenshot and synthetic Fitts trial windows.
- Rationale: some keyboard layouts/input methods may not deliver `Key_Q` consistently; checking the typed text makes Q-based aborts more reliable.

- Fixed the synthetic Fitts start-to-first-block transition.
- Rationale: after the participant clicked the synthetic Fitts start screen, the black instruction screen could remain visible above the block subprocess, making the experiment look stuck. The parent session now hides the transition screen once the block window has had time to appear and terminates the block subprocess if Q/Esc is pressed through the transition screen.

- Replaced the fixed synthetic Fitts reveal delay with a child-window readiness event.
- Rationale: hiding the black transition screen after a fixed delay could briefly expose the desktop before the Fitts block window was actually fullscreen. Each Fitts block now logs `window_shown`, and the parent session hides the transition screen only after seeing that event. The parent also shows a blank transition screen immediately when a block ends before displaying the pause page.

- Kept the fullscreen transition screen as a background layer during blocks instead of hiding it.
- Rationale: hiding the transition screen allowed the desktop to become visible when a block window closed before the next pause screen was drawn. The transition screen is now lowered behind the active experiment window, so it remains available as a black background between blocks without intercepting experiment input.

- Aligned synthetic Fitts Ninja Cursors pre-trial and active anchors with the physical screen center.
- Rationale: the eight Ninja cursors should start fixed at the screen center during the pre-trial phase, not at the synthetic Fitts blue home point.

- Stopped the synthetic Fitts task from recentering the native cursor on the blue home point during Ninja Cursors blocks.
- Rationale: the Fitts task's home-point cursor lock conflicted with Ninja Cursors' experiment-control anchor. For Ninja Cursors, the eight-cursor array is anchored at the physical screen center, matching the realistic screenshot task behavior; the synthetic blue home point remains part of the visual Fitts layout but is not used to force the native cursor during Ninja blocks.

- Added explicit Ninja Cursors instructions before Ninja blocks, including when Ninja is the first block.
- Rationale: participants need a reminder that the eight cursors start at the screen center, that the active cursor is shown in orange, and that gaze selects which cursor will be used.

## 2026-06-25

- Added a synthetic Fitts-with-distractors full-session protocol using generated targets and a FakeTargetFinder-style annotation source.
- Rationale: Julien Gori requested a classic Fitts-with-distractors comparison task for healthy participants, while reusing the existing Bubble Cursor, DynaSpot, Semantic Pointing, and Ninja Cursors implementations.

- Changed synthetic Fitts condition allocation from a fixed density per session to participant-level allocation over `ID x density` combinations.
- Rationale: Julien Gori advised not to treat Fitts ID and distractor density as full within-participant factors because this would make the experiment too long. The implementation now builds 24 combinations from 6 Fitts ID levels and 4 density levels, then assigns a subset per participant.

- Changed synthetic Fitts participant allocation so participants are grouped by the requested number of blocks.
- Rationale: with the default 12 synthetic blocks, P01 receives one half of the 24 ID-density combinations and P02 receives the other half; this improves coverage across participants while keeping each participant's session shorter.

- Added Balanced Latin Square ordering for synthetic Fitts blocks when the selected block count is even.
- Rationale: Julien Gori indicated that a BLS can be used for both the realistic and synthetic experiments to reduce order effects.

- Added a healthy-participant comparative protocol that runs both full sessions.
- Rationale: Julien Gori requested balancing the order between the synthetic Fitts task and the realistic screenshot task. Odd-numbered participants start with synthetic Fitts; even-numbered participants start with the realistic task.

- Moved the qualitative standard-mouse baseline tasks out of the tester technique list and into a separate first-page entry.
- Rationale: these three tasks are observational baseline tasks, not an interaction technique. Keeping them separate makes the panel clearer.

- Added a tester-only "Standard Mouse" mode with an optional pointer filter controlled by the existing filter selector.
- Rationale: Julien Gori suggested that the One Euro filter may need to be tested alone before deciding whether it should become an independent condition or an add-on to other techniques. Keeping the filter optional allows standard mouse without filtering, standard mouse with filtering, and later combinations with other techniques.

- Replaced the visible experiment-task-type dropdown with two explicit experiment protocol choices: `Patient protocol - realistic task` and `Control protocol - realistic task + synthetic Fitts`.
- Rationale: the experiment entry flow should match the actual study protocols. The standalone synthetic_fitts task remains an internal implementation path, while the participant-facing experiment mode now presents only complete protocols.

- Split controlled-experiment logs by population and task: patient realistic sessions now default to `patient_logs`, control realistic sessions in the comparative protocol go to `control_our_task`, and control synthetic Fitts sessions go to `control_fitts_synthetic`.
- Rationale: patient data and healthy-control comparison data must be separated at the directory level, while each session log now records its task type, log group, and output directory for traceability.

- Moved the healthy-control comparative controller log into its own `control_comparative` directory.
- Rationale: the top-level log that records the order of the two experiments, the between-task pause, and task-level return codes is not itself a realistic-task log or a synthetic-task log. Keeping it separate prevents `control_our_task` from receiving files when only the synthetic task has actually been completed.

- Renamed the participant-facing experiment choices to `Patient protocol - realistic task` and `Control protocol - realistic task + synthetic Fitts`, with French labels `Protocole patient - tâche réaliste` and `Protocole contrôle - tâche réaliste + Fitts synthétique`.
- Rationale: the previous labels `Complete experiment` and `Complete experiment + synthetic_fitts` did not clearly distinguish the patient protocol from the healthy-control comparative protocol.

- Fixed panel-to-command propagation for the Ninja Cursors debug status overlay in single-task, realistic-session, synthetic-session, and comparative-session launches.
- Rationale: the panel switch for the black gaze-tracking status bar should control whether `--ninja-hide-debug-status` is passed. It was previously forced hidden in several experiment paths.

- Made the panel refresh experiment-specific fields immediately when switching between the separated healthy-control tasks.
- Rationale: selecting the control realistic task or the control synthetic Fitts task should immediately update the visible parameter fields before pressing Start / Apply.

- Fixed the Ninja Cursors pre-trial anchor in the realistic screenshot task.
- Rationale: the realistic task now writes `ready x y` to the Ninja experiment-control file after the current screenshot has been loaded, using the same global start position as the trial cursor reset. This keeps the eight Ninja cursors centered on the trial start point before movement begins.

- Changed the experiment launch flow to use a dedicated protocol-selection page before the parameter page.
- Rationale: after selecting "Run an experiment", the user should first choose between the patient-style complete experiment and the control comparative protocol, then configure parameters on the next page.

- Kept the synthetic Fitts session screen visible between blocks instead of hiding it.
- Rationale: the desktop should never appear during an experimental session except when the session ends normally or is interrupted with Q/Esc.

- Passed the synthetic Fitts home-point coordinates to Ninja Cursors through the experiment-control file.
- Rationale: Ninja Cursors previously used its own anchor while the Fitts task reset the cursor to the blue home point, which could make the eight cursors jump at the beginning of a trial. The anchor is now aligned with the trial start point.

- Set protocol-dependent default trials per block: 8 for patient complete experiments and 12 for healthy-control comparative experiments, including the synthetic Fitts task.
- Rationale: patient sessions should be shorter, while healthy-control sessions should use the higher default requested for the full comparative protocol.

- Split the healthy-control comparative protocol at the panel level into two separately launched tasks: `control_our_task` and `control_fitts_synthetic`.
- Rationale: the realistic screenshot task and the synthetic Fitts-with-distractors task must be run and checked independently, while still belonging to the same healthy-control comparison protocol.

- Routed patient and healthy-control logs explicitly by task: patient complete experiments keep the default `patient_logs`, healthy-control realistic sessions use `control_our_task`, and healthy-control synthetic sessions use `control_fitts_synthetic`.
- Rationale: logs must be traceable to the correct participant group and task without mixing patient, control-realistic, and control-synthetic data.

- Added participant-ID-based default task selection for the separated healthy-control protocol.
- Rationale: odd-numbered participants are recommended to start with `control_fitts_synthetic`, while even-numbered participants are recommended to start with `control_our_task`, preserving the requested between-task counterbalancing even though the two tasks are launched separately.

- Ensured the controlled-experiment log roots exist in the project: `patient_logs`, `control_comparative`, `control_our_task`, and `control_fitts_synthetic`.
- Rationale: the panel and the file explorer should show the current official log destinations even before the first run creates a session subfolder.

- Added a second protocol-choice page after selecting the healthy-control comparative protocol.
- Rationale: during development, the two healthy-control tasks can still be tested separately; during actual data collection, the full comparative protocol can be launched as one ordered sequence from the participant ID.

- Added a between-task pause to the full healthy-control comparative session.
- Rationale: when the first experiment finishes, the participant sees which experiment just ended and which experiment will start next, then must click a button before the next experiment begins.

- Excluded `Slider` targets from Bubble Cursor absorption while keeping them in TargetFinder logs.
- Rationale: patient testing showed that scrollbars and sliders are unstable Bubble Cursor targets because many web scrollbars appear only during scrolling and their detected bounding boxes can distract the Bubble Cursor from nearby content. TargetFinder still detects and logs `Slider` targets, and direct clicks on detected sliders are logged as sliders, but Bubble Cursor no longer selects, highlights, or redirects clicks to `class_id=5` targets.

- Restored the intended Ninja Cursors trial state machine.
- Rationale: during the pre-trial countdown, the eight Ninja cursors should remain centered on the trial start point without any orange active cursor. The active orange cursor and native-cursor snap now start only after the countdown, and the realistic screenshot task writes `active x y` with the same start coordinates used during `ready`.

- Anchored Ninja Cursors to the physical screen center during realistic screenshot trials.
- Rationale: for the realistic complete experiment, Ninja Cursors should stay fixed at the screen center during countdown rather than at the scaled screenshot/image center. Synthetic Fitts keeps its own blue home-point anchor.

- Moved synthetic Fitts next-block details into the pause screen.
- Rationale: the session no longer shows an extra "Preparing block" page between Fitts blocks. The pause page now contains the completed/next block numbers and, for synthetic blocks, the next technique, ID, and distractor density.

- Made experiment start/pause screens use a local event loop with an early-click guard.
- Rationale: pressing the "Start/Continue" button, Q, or Esc on initialization and pause screens should always release the waiting screen. This avoids a race where a button click could be processed before the waiting loop started and leave the screen stuck.

- Restored Bubble Cursor absorption for `Slider` targets.
- Rationale: the previous patient-test workaround that excluded `class_id=5` from Bubble Cursor selection has been removed. Slider detections from TargetFinder are again treated like other Bubble Cursor targets, while the existing `b` key still toggles Bubble Cursor on and off.

- Changed the default experiment countdown to 0 seconds.
- Rationale: the control panel label, saved panel configuration, realistic task default, full realistic-session default, and synthetic Fitts default now all use a 0-second countdown unless explicitly changed in the panel.

- Made fullscreen instruction-screen continue buttons trigger on mouse press inside the button.
- Rationale: on macOS fullscreen/always-on-top instruction pages, the first click could be consumed by window activation or focus handling and the participant had to click the Fitts start button twice. Handling the button press directly makes the first click start the next phase.

- Added a guarded mouse-start path for fullscreen instruction screens.
- Rationale: after the participant presses the start button, the button is disabled, subsequent mouse press/release/double-click events are swallowed, and the next experimental phase starts only after the initiating mouse release has been consumed. This prevents repeated clicks on the start page from leaking into the first trial and causing an immediate failure.

- Moved the pause between the two healthy-control experiments into a standalone subprocess.
- Rationale: the synthetic Fitts session completed successfully, but the comparative parent process could crash while creating the Qt pause screen before launching the realistic task. Running the inter-task pause in a fresh process isolates Qt window lifecycle issues and lets the parent reliably continue to the second experiment.

- Disabled pending "Démarrage..." feedback on pause screens.
- Rationale: pause screens should remain stable rest screens. They still swallow the initiating mouse release to avoid leaking clicks into the next block or task, but the button text no longer changes to "Démarrage..." on pause.

- Removed extra continue-screen delay on pause screens.
- Rationale: the continue button no longer waits before releasing the pause screen, and synthetic Fitts clears the pause text immediately while the next block window appears.

- Made Semantic Pointing reset from a unique trial key and explicit start position.
- Rationale: Semantic Pointing's core movement algorithm is unchanged. The controlled tasks now write a unique `trial_key` and `start_position_global` into the annotation-control payload. Semantic Pointing reads those fields before checking whether the detector is active, so every trial, including repeated trial numbers across blocks, resets the white cursor and native pointer to the intended start point.

- Delayed synthetic Fitts target activation for Semantic Pointing until movement starts.
- Rationale: during countdown and release guard, the synthetic Fitts task keeps the cursor at the blue home point. For Semantic Pointing blocks, the synthetic target list is now exposed to the technique only when the movement phase starts, avoiding pre-movement drift while preserving the original Semantic Pointing speed-control algorithm.
