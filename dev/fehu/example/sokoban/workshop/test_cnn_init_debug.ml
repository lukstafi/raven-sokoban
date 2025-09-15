(*
Test the initialize_cnn_policy and initialize_cnn_value functions
*)

open Workshop.Exercise4_cnn
open Workshop.Slide6_cnn

let test_init_functions () =
  Printf.printf "Testing initialization functions...\n\n";

  (* Test 1: Initialize with grid_size=10 *)
  Printf.printf "Test 1: Testing initialize_cnn_policy with grid_size=10...\n";
  let policy_net, policy_params = try
    initialize_cnn_policy ~grid_size:10 ()
  with e ->
    Printf.printf "  ERROR: %s\n" (Printexc.to_string e);
    Printf.printf "  Backtrace:\n%s\n" (Printexc.get_backtrace ());
    raise e
  in
  Printf.printf "  Policy initialized successfully\n";

  (* Test 2: Initialize value network *)
  Printf.printf "\nTest 2: Testing initialize_cnn_value with grid_size=10...\n";
  let value_net, value_params = try
    initialize_cnn_value ~grid_size:10 ()
  with e ->
    Printf.printf "  ERROR: %s\n" (Printexc.to_string e);
    Printf.printf "  Backtrace:\n%s\n" (Printexc.get_backtrace ());
    raise e
  in
  Printf.printf "  Value initialized successfully\n";

  (* Test 3: Apply networks *)
  Printf.printf "\nTest 3: Testing forward pass with initialized networks...\n";
  let input = Rune.randn Rune.c Rune.float32 [|1; 1; 10; 10|] in

  let policy_output = try
    Kaun.apply policy_net policy_params ~training:false input
  with e ->
    Printf.printf "  ERROR in policy forward: %s\n" (Printexc.to_string e);
    raise e
  in
  Printf.printf "  Policy output shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Rune.shape policy_output))));

  let value_output = try
    Kaun.apply value_net value_params ~training:false input
  with e ->
    Printf.printf "  ERROR in value forward: %s\n" (Printexc.to_string e);
    raise e
  in
  Printf.printf "  Value output shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Rune.shape value_output))));

  (* Test 4: Test prepare_states_batch_cnn *)
  Printf.printf "\nTest 4: Testing prepare_states_batch_cnn...\n";
  let states = Array.init 5 (fun _ ->
    Rune.randn Rune.c Rune.float32 [|10; 10|]
  ) in
  let batch = try
    prepare_states_batch_cnn states
  with e ->
    Printf.printf "  ERROR: %s\n" (Printexc.to_string e);
    raise e
  in
  Printf.printf "  Batch shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Rune.shape batch))));

  (* Test 5: Apply to batch *)
  Printf.printf "\nTest 5: Testing network on prepared batch...\n";
  let policy_batch_output = try
    Kaun.apply policy_net policy_params ~training:false batch
  with e ->
    Printf.printf "  ERROR in batch policy forward: %s\n" (Printexc.to_string e);
    raise e
  in
  Printf.printf "  Policy batch output shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Rune.shape policy_batch_output))));

  Printf.printf "\nAll initialization tests passed!\n"


let test_with_environment () =
  Printf.printf "\n\nTesting with actual environment...\n\n";

  (* Create a verified-curriculum environment *)
  let env = Workshop.Verified.sokoban_curriculum ~max_steps:200 () in

  (* Initialize networks *)
  let policy_net, policy_params = initialize_cnn_policy ~grid_size:10 () in

  (* Test 1: Reset environment *)
  Printf.printf "Test 1: Resetting environment...\n";
  let state, _ = env.Fehu.Env.reset () in
  Printf.printf "  State shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Rune.shape state))));

  (* Test 2: Prepare state for CNN *)
  Printf.printf "\nTest 2: Preparing state for CNN...\n";
  let state_cnn = Rune.reshape [|1; 1; 10; 10|] state in
  Printf.printf "  CNN state shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Rune.shape state_cnn))));

  (* Test 3: Apply policy *)
  Printf.printf "\nTest 3: Applying policy network...\n";
  let logits = try
    Kaun.apply policy_net policy_params ~training:false state_cnn
  with e ->
    Printf.printf "  ERROR: %s\n" (Printexc.to_string e);
    Printf.printf "  Backtrace:\n%s\n" (Printexc.get_backtrace ());
    raise e
  in
  Printf.printf "  Logits shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Rune.shape logits))));

  (* Test 4: Try collecting an episode *)
  Printf.printf "\nTest 4: Collecting episode...\n";
  let episode = try
    collect_episode_cnn env policy_net policy_params 10
  with e ->
    Printf.printf "  ERROR: %s\n" (Printexc.to_string e);
    Printf.printf "  Backtrace:\n%s\n" (Printexc.get_backtrace ());
    raise e
  in
  Printf.printf "  Episode collected: %d states, %d actions\n"
    (Array.length episode.Workshop.Slide3.states) (Array.length episode.Workshop.Slide3.actions);

  Printf.printf "\nEnvironment tests passed!\n"


let test_full_training_step () =
  Printf.printf "\n\nTesting full training step...\n\n";

  (* Create environment *)
  let env = Workshop.Verified.sokoban_curriculum ~max_steps:200 () in

  (* Run one episode of A2C training *)
  Printf.printf "Running train_actor_critic_cnn for 1 episode...\n";
  let history = try
    train_actor_critic_cnn env 1 0.001 0.003 0.99 ~grid_size:10 ()
  with e ->
    Printf.printf "  ERROR: %s\n" (Printexc.to_string e);
    Printf.printf "  Backtrace:\n%s\n" (Printexc.get_backtrace ());
    raise e
  in
  Printf.printf "  Training completed!\n";
  Printf.printf "  Returns: %.3f\n" history.Workshop.Slide4.returns.(0);
  Printf.printf "  Losses: %.6f\n" history.Workshop.Slide4.losses.(0);

  Printf.printf "\nFull training test passed!\n"


let () =
  Printexc.record_backtrace true;
  test_init_functions ();
  test_with_environment ();
  test_full_training_step ()