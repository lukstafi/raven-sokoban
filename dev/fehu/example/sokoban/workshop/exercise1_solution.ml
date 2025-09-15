(*
Exercise 1 Solution: Batched REINFORCE for Processing All States

This solution processes all states in an episode by batching them together
in a single forward pass, avoiding the indexing issues during autodiff.
*)

open Slide2
open Slide3
open Slide4

(* Replace the loss computation in slide4.ml (lines 43-106) with this: *)
let compute_batched_loss episode_data returns params policy_net _grid_size =
  let device = Rune.c in
  let n_actions = Array.length episode_data.actions in

  if n_actions = 0 then
    Rune.scalar device Rune.float32 0.0
  else
    (* Step 1: Stack all states that have corresponding actions *)
    (* Important: We only take states up to n_actions to avoid shape mismatch *)
    let states_list =
      Array.to_list (Array.sub episode_data.states 0 n_actions) in
    let all_states = Rune.stack ~axis:0 states_list in

    (* Step 2: Process all states at once through the network *)
    let all_logits =
      Kaun.apply policy_net params ~training:true all_states in

    (* Step 3: Compute log probabilities *)
    let all_log_probs = log_softmax ~axis:(-1) all_logits in

    (* Step 4: Convert actions to one-hot encodings *)
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

    (* Step 5: Select log probs using element-wise multiply and sum *)
    let selected_log_probs =
      Rune.sum ~axes:[|1|] (Rune.mul all_actions_one_hot all_log_probs) in

    (* Step 6: Weight by returns and compute mean loss *)
    let returns_tensor =
      Rune.init device Rune.float32 [|n_actions|] (fun idxs ->
        returns.(idxs.(0))
      ) in

    let weighted_losses =
      Rune.mul (Rune.neg selected_log_probs) returns_tensor in

    Rune.mean weighted_losses

(* Complete training function with batched loss *)
let train_reinforce_batched env n_episodes learning_rate gamma ?(grid_size=5) () =
  (* Initialize policy *)
  let policy_net, params = initialize_policy ~grid_size () in

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

    (* Compute policy gradient loss using batched approach *)
    let loss, grads = Kaun.value_and_grad (fun p ->
      compute_batched_loss episode_data returns p policy_net grid_size
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
    let n_actions = Array.length episode_data.actions in
    if episode = 1 || episode mod 10 = 0 || episode = n_episodes then
      Printf.printf "Episode %4d | Return: %7.2f | Loss: %7.4f | States processed: %d\n%!"
        episode total_return (Rune.item [] loss) n_actions
  done;

  (* Return training results *)
  {
    returns = history_returns;
    losses = history_losses;
    collected_episodes = List.rev !collected_episodes;
  }

(*
Key improvements over the original implementation:

1. **Processes ALL states**: Instead of limiting to 10 states, we now process
   all states in the episode (typically 20-100 states).

2. **Batched computation**: All states are stacked into a single tensor and
   processed in one forward pass, avoiding the indexing issue.

3. **Shape alignment fix**: We only take states up to n_actions to ensure
   shapes match between states and actions arrays.

4. **Efficient action selection**: Uses one-hot encoding and element-wise
   multiplication to select action log probabilities without indexing.

Performance impact:
- Before: Only 10 states per episode were used for learning
- After: All states (20-100+) are used, leading to:
  * 2-10x more gradient information per episode
  * Faster convergence (fewer episodes to reach goal)
  * Better sample efficiency
  * More stable learning curves
*)