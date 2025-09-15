(*
Batched REINFORCE implementation that processes all states at once
to avoid indexing issues during autodiff
*)
open Slide2
open Slide3
open Slide4  (* For training_history type *)

let train_reinforce env n_episodes learning_rate gamma ?(grid_size=5) () =
  (* Initialize policy *)
  let policy_net, params = initialize_policy ~grid_size () in
  let device = Rune.c in

  (* Create optimizer *)
  let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
  let opt_state = ref (optimizer.init params) in

  (* Storage for visualization *)
  let collected_episodes = ref [] in

  (* History tracking *)
  let history_returns = Array.make n_episodes 0.0 in
  let history_losses = Array.make n_episodes 0.0 in

  (* Training loop *)
  for episode = 1 to n_episodes do
    (* Collect episode *)
    let episode_data =
      collect_episode env policy_net params 100 in

    (* Store selected episodes *)
    if episode mod (n_episodes / 10) = 0 || episode = n_episodes then
      collected_episodes := episode_data :: !collected_episodes;

    (* Compute returns *)
    let returns = compute_returns episode_data.rewards gamma in

    (* Get actual episode length *)
    let n_actions = Array.length episode_data.actions in

    (* Compute policy gradient loss using batched approach *)
    let loss, grads = Kaun.value_and_grad (fun p ->
      if n_actions = 0 then
        Rune.scalar device Rune.float32 0.0
      else
        (* Stack all states into a batch *)
        (* We need to process only the states that have corresponding actions *)
        let states_list =
          Array.to_list (Array.sub episode_data.states 0 n_actions) in
        (* Each state is [grid_size; grid_size], we need [batch; grid_size; grid_size] *)
        let all_states =
          if List.length states_list > 0 then
            (* Stack expects [grid_size; grid_size] shaped tensors *)
            Rune.stack ~axis:0 states_list
          else
            (* Empty batch case *)
            Rune.zeros device Rune.float32 [|1; grid_size; grid_size|]
        in

        (* Process all states at once *)
        let all_logits =
          Kaun.apply policy_net p ~training:true all_states in

        (* Compute log probabilities for all states *)
        let all_log_probs = log_softmax ~axis:(-1) all_logits in

        (* Create action indices tensor for gathering *)
        let action_indices =
          Array.init n_actions (fun i ->
            let action_tensor = episode_data.actions.(i) in
            Rune.astype Rune.int32 action_tensor
          ) in

        (* Create one-hot encodings for all actions at once *)
        let actions_one_hot_list =
          List.mapi (fun _i action_idx ->
            let one_hot = Rune.one_hot ~num_classes:4 action_idx in
            (* one_hot is shape [4], convert to float32 *)
            Rune.astype Rune.float32 one_hot
          ) (Array.to_list action_indices) in

        (* Stack all one-hot actions *)
        let all_actions_one_hot =
          if List.length actions_one_hot_list > 0 then
            Rune.stack ~axis:0 actions_one_hot_list
          else
            Rune.zeros device Rune.float32 [|1; 4|]
        in

        (* Select log probs using element-wise multiply and sum along action dimension *)
        (* all_log_probs is [n_actions, 4], all_actions_one_hot is [n_actions, 4] *)
        let selected_log_probs =
          Rune.sum ~axes:[|1|] (Rune.mul all_actions_one_hot all_log_probs) in

        (* Create returns tensor *)
        let returns_tensor =
          Rune.init device Rune.float32 [|n_actions|] (fun idxs ->
            returns.(idxs.(0))
          ) in

        (* Compute weighted loss: -log_prob * return *)
        let weighted_losses =
          Rune.mul (Rune.neg selected_log_probs) returns_tensor in

        (* Average loss across all states *)
        Rune.mean weighted_losses
    ) params in

    (* Update parameters *)
    let updates, new_state =
      optimizer.update !opt_state params grads in
    opt_state := new_state;
    Kaun.Optimizer.apply_updates_inplace params updates;

    (* Store history *)
    let total_return = Array.fold_left (+.) 0.0 episode_data.rewards in
    history_returns.(episode - 1) <- total_return;
    history_losses.(episode - 1) <- Rune.item [] loss;

    (* Print progress *)
    if episode = 1 || episode mod 10 = 0 || episode = n_episodes then
      Printf.printf "Episode %4d | Return: %7.2f | Loss: %7.4f | States processed: %d\n%!"
        episode total_return (Rune.item [] loss) n_actions
  done;

  (* Return in the expected format *)
  (policy_net, params, List.rev !collected_episodes,
   { returns = history_returns;
     losses = history_losses;
     collected_episodes = List.rev !collected_episodes })