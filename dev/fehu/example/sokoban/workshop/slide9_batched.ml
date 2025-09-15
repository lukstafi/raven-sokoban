(*
Batched PPO (Proximal Policy Optimization) implementation
Processes all states in episodes using batched operations
*)
open Slide2
open Slide3
open Slide4  (* For training_history type *)

let train_reinforce_plus_plus env n_episodes learning_rate gamma epsilon beta ?(grid_size=5) () =
  let baseline_alpha = 0.95 in
  let ppo_epochs = 4 in
  (* Initialize policy *)
  let policy_net, params = initialize_policy ~grid_size () in
  let device = Rune.c in

  (* Create optimizer *)
  let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
  let opt_state = ref (optimizer.init params) in

  (* Baseline for variance reduction *)
  let baseline = ref 0.0 in

  (* Storage for visualization *)
  let collected_episodes = ref [] in
  let history_returns = Array.make n_episodes 0.0 in
  let history_losses = Array.make n_episodes 0.0 in
  let history_kl = Array.make n_episodes 0.0 in

  (* Parameters for old policy (used for importance sampling) *)
  let old_params = ref params in

  for episode = 1 to n_episodes do
    (* Collect episode *)
    let episode_data = collect_episode env policy_net params 100 in

    (* Store selected episodes *)
    if episode mod (n_episodes / 10) = 0 || episode = n_episodes then
      collected_episodes := episode_data :: !collected_episodes;

    (* Compute returns *)
    let returns = compute_returns episode_data.rewards gamma in
    let n_actions = Array.length episode_data.actions in

    if n_actions > 0 then begin
      (* Update baseline *)
      let episode_return = Array.fold_left (+.) 0.0 episode_data.rewards in
      baseline := !baseline *. (1.0 -. baseline_alpha) +.
                  episode_return *. baseline_alpha;

      (* Compute advantages *)
      let advantages = Array.map (fun r -> r -. !baseline) returns in

      (* Batch all states that have actions *)
      let states_list = Array.to_list (Array.sub episode_data.states 0 n_actions) in
      let all_states = Rune.stack ~axis:0 states_list in

      (* Compute old log probs for all states at once *)
      let old_all_logits =
        Kaun.apply policy_net !old_params ~training:false all_states in
      let old_all_log_probs = log_softmax ~axis:(-1) old_all_logits in

      (* Extract old log probs for taken actions *)
      let action_indices = Array.init n_actions (fun i ->
        Rune.astype Rune.int32 episode_data.actions.(i)
      ) in

      let old_log_probs_array = Array.mapi (fun i _action_idx ->
        let action_int = int_of_float (Rune.item [] episode_data.actions.(i)) in
        let action_int = max 0 (min 3 action_int) in
        Rune.item [i; action_int] old_all_log_probs
      ) action_indices in

      (* PPO update with multiple epochs *)
      let final_loss = ref (Rune.scalar device Rune.float32 0.0) in
      let final_kl = ref 0.0 in

      for _ppo_epoch = 1 to ppo_epochs do
        (* Compute policy gradient with clipping *)
        let loss, grads = Kaun.value_and_grad (fun p ->
          (* Get current policy for all states *)
          let all_logits = Kaun.apply policy_net p ~training:true all_states in
          let all_log_probs = log_softmax ~axis:(-1) all_logits in
          let all_probs = Rune.exp all_log_probs in

          (* Convert actions to one-hot *)
          let actions_one_hot_list = List.map (fun action_idx ->
            let one_hot = Rune.one_hot ~num_classes:4 action_idx in
            Rune.astype Rune.float32 one_hot
          ) (Array.to_list action_indices) in
          let all_actions_one_hot = Rune.stack ~axis:0 actions_one_hot_list in

          (* Select log probs for taken actions *)
          let new_log_probs =
            Rune.sum ~axes:[|1|] (Rune.mul all_actions_one_hot all_log_probs) in

          (* Create old log probs tensor *)
          let old_log_probs_tensor =
            Rune.init device Rune.float32 [|n_actions|] (fun idxs ->
              old_log_probs_array.(idxs.(0))
            ) in

          (* Compute ratios *)
          let log_ratios = Rune.sub new_log_probs old_log_probs_tensor in
          let ratios = Rune.exp log_ratios in

          (* Clip ratios *)
          let clipped_ratios = Rune.minimum
            (Rune.maximum ratios
              (Rune.scalar device Rune.float32 (1.0 -. epsilon)))
            (Rune.scalar device Rune.float32 (1.0 +. epsilon)) in

          (* Advantages tensor *)
          let advantages_tensor =
            Rune.init device Rune.float32 [|n_actions|] (fun idxs ->
              advantages.(idxs.(0))
            ) in

          (* Compute objectives *)
          let obj1 = Rune.mul ratios advantages_tensor in
          let obj2 = Rune.mul clipped_ratios advantages_tensor in
          let clipped_objectives = Rune.minimum obj1 obj2 in

          (* Policy loss (negative for gradient ascent) *)
          let policy_loss = Rune.neg (Rune.mean clipped_objectives) in

          (* KL divergence penalty *)
          let old_probs = Rune.exp old_all_log_probs in
          let kl_divs = Rune.sum ~axes:[|1|]
            (Rune.mul old_probs
              (Rune.sub (Rune.log (Rune.add old_probs (Rune.scalar device Rune.float32 1e-8)))
                        (Rune.log (Rune.add all_probs (Rune.scalar device Rune.float32 1e-8))))) in
          let kl_penalty = Rune.mul
            (Rune.scalar device Rune.float32 beta)
            (Rune.mean kl_divs) in

          (* Store KL for monitoring *)
          final_kl := Rune.item [] (Rune.mean kl_divs);

          (* Total loss *)
          Rune.add policy_loss kl_penalty
        ) params in

        final_loss := loss;

        (* Update parameters *)
        let updates, new_state =
          optimizer.update !opt_state params grads in
        opt_state := new_state;
        Kaun.Optimizer.apply_updates_inplace params updates
      done;

      (* Update old parameters for next episode *)
      old_params := params;

      (* Store history *)
      history_returns.(episode - 1) <- episode_return;
      history_losses.(episode - 1) <- Rune.item [] !final_loss;
      history_kl.(episode - 1) <- !final_kl;

      (* Print progress *)
      if episode = 1 || episode mod 10 = 0 || episode = n_episodes then
        Printf.printf "Episode %4d | Return: %7.2f | Loss: %7.4f | KL: %7.5f | States: %d\n%!"
          episode episode_return
          (Rune.item [] !final_loss)
          !final_kl
          n_actions
    end
  done;

  (* Return results in expected format *)
  (policy_net, params,
   { returns = history_returns;
     losses = history_losses;
     collected_episodes = List.rev !collected_episodes })