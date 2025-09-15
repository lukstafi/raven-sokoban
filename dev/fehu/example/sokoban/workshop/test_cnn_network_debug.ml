(*
Test the actual CNN network initialization and application
*)

open Workshop.Exercise4_cnn

let test_cnn_network () =
  let device = Rune.c in

  (* Test with grid size 10 (verified-curriculum) *)
  Printf.printf "Testing CNN network with 10x10 grid...\n\n";

  (* Test 1: Create policy network *)
  Printf.printf "Test 1: Creating CNN policy network for 10x10...\n";
  let policy_net = create_cnn_policy_network_for_grid 4 10 in
  Printf.printf "  Policy network created\n";

  (* Test 2: Initialize policy network *)
  Printf.printf "\nTest 2: Initializing policy network...\n";
  let rng = Rune.Rng.key 42 in
  let policy_params = try
    Kaun.init policy_net ~rngs:rng ~device ~dtype:Rune.float32
  with e ->
    Printf.printf "  ERROR during policy init: %s\n" (Printexc.to_string e);
    Printf.printf "  Backtrace:\n%s\n" (Printexc.get_backtrace ());
    raise e
  in
  Printf.printf "  Policy network initialized\n";

  (* Test 3: Create value network *)
  Printf.printf "\nTest 3: Creating CNN value network for 10x10...\n";
  let value_net = create_cnn_value_network_for_grid 10 in
  Printf.printf "  Value network created\n";

  (* Test 4: Initialize value network *)
  Printf.printf "\nTest 4: Initializing value network...\n";
  let value_params = try
    Kaun.init value_net ~rngs:rng ~device ~dtype:Rune.float32
  with e ->
    Printf.printf "  ERROR during value init: %s\n" (Printexc.to_string e);
    Printf.printf "  Backtrace:\n%s\n" (Printexc.get_backtrace ());
    raise e
  in
  Printf.printf "  Value network initialized\n";

  (* Test 5: Apply networks to a sample input *)
  Printf.printf "\nTest 5: Testing forward pass...\n";
  let input = Rune.randn device Rune.float32 [|1; 1; 10; 10|] in
  Printf.printf "  Input shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Rune.shape input))));

  let policy_output = try
    Kaun.apply policy_net policy_params ~training:false input
  with e ->
    Printf.printf "  ERROR during policy forward: %s\n" (Printexc.to_string e);
    Printf.printf "  Backtrace:\n%s\n" (Printexc.get_backtrace ());
    raise e
  in
  Printf.printf "  Policy output shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Rune.shape policy_output))));

  let value_output = try
    Kaun.apply value_net value_params ~training:false input
  with e ->
    Printf.printf "  ERROR during value forward: %s\n" (Printexc.to_string e);
    Printf.printf "  Backtrace:\n%s\n" (Printexc.get_backtrace ());
    raise e
  in
  Printf.printf "  Value output shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Rune.shape value_output))));

  (* Test 6: Test with batch *)
  Printf.printf "\nTest 6: Testing with batch of 32...\n";
  let batch_input = Rune.randn device Rune.float32 [|32; 1; 10; 10|] in

  let batch_policy_output = try
    Kaun.apply policy_net policy_params ~training:false batch_input
  with e ->
    Printf.printf "  ERROR during batch policy forward: %s\n" (Printexc.to_string e);
    raise e
  in
  Printf.printf "  Batch policy output shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Rune.shape batch_policy_output))));

  let batch_value_output = try
    Kaun.apply value_net value_params ~training:false batch_input
  with e ->
    Printf.printf "  ERROR during batch value forward: %s\n" (Printexc.to_string e);
    raise e
  in
  Printf.printf "  Batch value output shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Rune.shape batch_value_output))));

  Printf.printf "\nAll network tests passed!\n"

let () =
  Printexc.record_backtrace true;
  test_cnn_network ()