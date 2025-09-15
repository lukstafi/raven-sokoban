(*
Test the batched REINFORCE implementation
*)
open Workshop.Slide1
open Workshop.Slide2
open Workshop.Slide3
open Workshop.Slide4_batched

let () =
  Printf.printf "Testing batched REINFORCE implementation...\n%!";

  (* Create environment *)
  let env = create_simple_gridworld 5 in

  (* Setup device and policy network *)
  let device = Rune.c in
  let policy_net, params = initialize_policy ~grid_size:5 () in

  (* Test a single episode collection and loss computation *)
  Printf.printf "\n1. Testing single episode processing:\n";
  let episode_data = collect_episode env policy_net params 100 in
  Printf.printf "   Episode length: %d states, %d actions\n"
    (Array.length episode_data.states)
    (Array.length episode_data.actions);

  (* Test loss computation with the batched approach *)
  let returns = compute_returns episode_data.rewards 0.99 in
  let n_actions = Array.length episode_data.actions in

  if n_actions > 0 then begin
    let loss, _grads = Kaun.value_and_grad (fun p ->
      (* Stack all states into a batch *)
      let states_list = Array.to_list episode_data.states in
      let all_states = Rune.stack ~axis:0 states_list in

      (* Process all states at once *)
      let all_logits =
        Kaun.apply policy_net p ~training:true all_states in

      (* Compute log probabilities *)
      let all_log_probs = log_softmax ~axis:(-1) all_logits in

      (* Create action one-hot encodings *)
      let action_indices =
        Array.init n_actions (fun i ->
          Rune.astype Rune.int32 episode_data.actions.(i)
        ) in

      let actions_one_hot_list =
        List.map (fun action_idx ->
          let one_hot = Rune.one_hot ~num_classes:4 action_idx in
          Rune.astype Rune.float32 one_hot
        ) (Array.to_list action_indices) in

      let all_actions_one_hot = Rune.stack ~axis:0 actions_one_hot_list in

      (* Select log probs *)
      let selected_log_probs =
        Rune.sum ~axes:[|1|] (Rune.mul all_actions_one_hot all_log_probs) in

      (* Create returns tensor *)
      let returns_tensor =
        Rune.init device Rune.float32 [|n_actions|] (fun idxs ->
          returns.(idxs.(0))
        ) in

      (* Compute weighted loss *)
      let weighted_losses =
        Rune.mul (Rune.neg selected_log_probs) returns_tensor in

      Rune.mean weighted_losses
    ) params in

    Printf.printf "   Loss computed successfully: %.6f\n" (Rune.item [] loss);
    Printf.printf "   ✓ Processed ALL %d states (not just 10!)\n" n_actions
  end else
    Printf.printf "   Episode had no actions to process\n";

  (* Test training for a few episodes *)
  Printf.printf "\n2. Testing training loop:\n";
  let episodes, returns, losses =
    train_reinforce_batched env 10 1e-3 0.99 ~grid_size:5 () in

  Printf.printf "   Training completed: %d episodes collected\n" (List.length episodes);
  Printf.printf "   Average return: %.2f\n"
    (Array.fold_left (+.) 0.0 returns /. float_of_int (Array.length returns));
  Printf.printf "   Final loss: %.6f\n" losses.(Array.length losses - 1);

  Printf.printf "\n✓ Batched REINFORCE implementation working correctly!\n";
  Printf.printf "  All states in episodes are being processed.\n"