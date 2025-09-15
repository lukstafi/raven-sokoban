(*
Minimal test to debug the conv2d 4D tensor error
*)

let test_kaun_conv2d () =
  let device = Rune.c in

  (* Test 1: Create a simple conv2d layer *)
  Printf.printf "Test 1: Creating conv2d layer...\n";
  let conv_layer = Kaun.Layer.conv2d ~in_channels:1 ~out_channels:16
                                     ~kernel_size:(3, 3) () in
  Printf.printf "  Conv2d layer created\n";

  (* Test 2: Initialize the layer *)
  Printf.printf "\nTest 2: Initializing conv2d layer...\n";
  let rng = Rune.Rng.key 42 in
  let params = try
    Kaun.init conv_layer ~rngs:rng ~device ~dtype:Rune.float32
  with e ->
    Printf.printf "  ERROR during init: %s\n" (Printexc.to_string e);
    raise e
  in
  Printf.printf "  Conv2d layer initialized\n";

  (* Test 3: Create a 4D input tensor *)
  Printf.printf "\nTest 3: Creating 4D input tensor...\n";
  let input = Rune.randn device Rune.float32 [|1; 1; 10; 10|] in
  Printf.printf "  Input shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Rune.shape input))));

  (* Test 4: Apply the layer *)
  Printf.printf "\nTest 4: Applying conv2d layer...\n";
  let output = try
    Kaun.apply conv_layer params ~training:false input
  with e ->
    Printf.printf "  ERROR during apply: %s\n" (Printexc.to_string e);
    raise e
  in
  Printf.printf "  Output shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Rune.shape output))));

  (* Test 5: Test with a sequential network *)
  Printf.printf "\nTest 5: Testing sequential network...\n";
  let net = Kaun.Layer.sequential [
    Kaun.Layer.conv2d ~in_channels:1 ~out_channels:16 ~kernel_size:(3, 3) ();
    Kaun.Layer.relu ();
  ] in

  let net_params = try
    Kaun.init net ~rngs:rng ~device ~dtype:Rune.float32
  with e ->
    Printf.printf "  ERROR during sequential init: %s\n" (Printexc.to_string e);
    raise e
  in

  let net_output = try
    Kaun.apply net net_params ~training:false input
  with e ->
    Printf.printf "  ERROR during sequential apply: %s\n" (Printexc.to_string e);
    raise e
  in
  Printf.printf "  Sequential output shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Rune.shape net_output))));

  Printf.printf "\nAll tests passed!\n"

let () = test_kaun_conv2d ()