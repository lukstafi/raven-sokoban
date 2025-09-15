(*
CNN-based PPO (Proximal Policy Optimization) implementation
Uses convolutional neural networks for improved spatial understanding
*)
open Slide3  (* For compute_returns *)
open Slide4  (* For training_history type *)
open Exercise4_cnn

let train_reinforce_plus_plus_cnn env n_episodes learning_rate gamma epsilon beta ?(grid_size=5) () =
  (* Initialize CNN policy network *)
  let policy_net, policy_params = initialize_cnn_policy ~grid_size () in

  let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
  let opt_state = ref (optimizer.init policy_params) in

  (* Storage for episodes *)
  let collected_episodes = ref [] in
  let history_returns = Array.make n_episodes 0.0 in
  let history_actor_losses = Array.make n_episodes 0.0 in
  let history_critic_losses = Array.make n_episodes 0.0 in  (* PPO doesn't use critic but kept for compatibility *)

  for episode = 1 to n_episodes do
    (* Collect episode with CNN-ready states *)
    let episode_data = collect_episode_cnn env policy_net policy_params 100 in

    (* Store selected episodes *)
    let store_freq = max 1 (n_episodes / 10) in
    if episode mod store_freq = 0 || episode = n_episodes then
      collected_episodes := episode_data :: !collected_episodes;

    let returns = compute_returns episode_data.rewards gamma in
    let n_actions = Array.length episode_data.actions in

    if n_actions > 0 then begin
      (* Prepare states for CNN: [batch_size, 1, 5, 5] *)
      let all_states = prepare_states_batch_cnn (Array.sub episode_data.states 0 n_actions) in

      (* Compute old probabilities for PPO *)
      let old_logits = Kaun.apply policy_net policy_params ~training:false all_states in
      let old_log_probs_all = Slide2.log_softmax ~axis:(-1) old_logits in

      (* Get old log probs for taken actions *)
      let action_indices = Array.init n_actions (fun i ->
        Rune.astype Rune.int32 episode_data.actions.(i)
      ) in

      let actions_one_hot_list = List.map (fun action_idx ->
        let one_hot = Rune.one_hot ~num_classes:4 action_idx in
        Rune.astype Rune.float32 one_hot
      ) (Array.to_list action_indices) in

      let all_actions_one_hot = Rune.stack ~axis:0 actions_one_hot_list in

      let old_log_probs =
        Rune.sum ~axes:[|1|] (Rune.mul all_actions_one_hot old_log_probs_all) in

      (* PPO update with clipping *)
      let loss, grads = Kaun.value_and_grad (fun pp ->
        (* Get new log probabilities *)
        let new_logits = Kaun.apply policy_net pp ~training:true all_states in
        let new_log_probs_all = Slide2.log_softmax ~axis:(-1) new_logits in
        let new_log_probs =
          Rune.sum ~axes:[|1|] (Rune.mul all_actions_one_hot new_log_probs_all) in

        (* Compute probability ratio *)
        let log_ratio = Rune.sub new_log_probs old_log_probs in
        let ratio = Rune.exp log_ratio in

        (* Create returns tensor *)
        let returns_tensor = Rune.init Rune.c Rune.float32 [|n_actions|] (fun idxs ->
          returns.(idxs.(0))
        ) in

        (* Clipped surrogate objective *)
        let unclipped = Rune.mul ratio returns_tensor in
        let clipped = Rune.mul
          (Rune.clip ratio
            ~min:(1.0 -. epsilon)
            ~max:(1.0 +. epsilon))
          returns_tensor in
        let surrogate = Rune.minimum unclipped clipped in

        (* KL penalty *)
        let kl_div = Rune.mean (Rune.sub old_log_probs new_log_probs) in

        (* Combined loss: negative surrogate + KL penalty *)
        Rune.add (Rune.neg (Rune.mean surrogate)) (Rune.mul (Rune.scalar Rune.c Rune.float32 beta) kl_div)
      ) policy_params in

      (* Apply gradients *)
      let updates, new_opt_state = optimizer.update !opt_state policy_params grads in
      opt_state := new_opt_state;
      Kaun.Optimizer.apply_updates_inplace policy_params updates;

      (* Record metrics *)
      history_returns.(episode - 1) <- Array.fold_left (+.) 0.0 episode_data.rewards;
      history_actor_losses.(episode - 1) <- Rune.item [] loss;
      history_critic_losses.(episode - 1) <- 0.0;  (* No critic in basic PPO *)
    end else begin
      (* Empty episode *)
      history_returns.(episode - 1) <- 0.0;
      history_actor_losses.(episode - 1) <- 0.0;
      history_critic_losses.(episode - 1) <- 0.0;
    end
  done;

  (* Return training history *)
  {
    returns = history_returns;
    losses = history_actor_losses;  (* Policy losses for PPO *)
    collected_episodes = List.rev !collected_episodes;
  }