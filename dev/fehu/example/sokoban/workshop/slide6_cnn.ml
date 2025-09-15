(*
CNN-based A2C (Advantage Actor-Critic) implementation
Uses convolutional neural networks for both policy and value functions
*)
open Slide3  (* For compute_returns *)
open Slide4  (* For training_history type *)
open Exercise4_cnn

let train_actor_critic_cnn env n_episodes lr_actor lr_critic gamma ?(grid_size=5) () =
  (* Initialize CNN networks *)
  let policy_net, policy_params = initialize_cnn_policy ~grid_size () in
  let value_net, value_params = initialize_cnn_value ~grid_size () in

  (* Separate optimizers for actor and critic *)
  let policy_opt = Kaun.Optimizer.adam ~lr:lr_actor () in
  let value_opt = Kaun.Optimizer.adam ~lr:lr_critic () in
  let policy_opt_state = ref (policy_opt.init policy_params) in
  let value_opt_state = ref (value_opt.init value_params) in

  (* Storage for episodes *)
  let collected_episodes = ref [] in
  let history_returns = Array.make n_episodes 0.0 in
  let history_actor_losses = Array.make n_episodes 0.0 in
  let history_critic_losses = Array.make n_episodes 0.0 in

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

      (* Compute all value estimates in one pass *)
      let all_values = Kaun.apply value_net value_params ~training:false all_states in
      let values_array = Array.init n_actions (fun i ->
        Rune.item [i] all_values
      ) in

      (* Compute advantages *)
      let advantages = Array.mapi (fun i r -> r -. values_array.(i)) returns in

      (* Update critic with batched computation *)
      let value_loss, value_grads = Kaun.value_and_grad (fun vp ->
        (* Compute all predictions at once *)
        let pred_tensor = Kaun.apply value_net vp ~training:true all_states in
        let pred_squeezed = Rune.squeeze ~axes:[|1|] pred_tensor in
        let returns_tensor = Rune.create Rune.c Rune.float32 [|n_actions|] returns in
        Kaun.Loss.mse pred_squeezed returns_tensor
      ) value_params in

      let value_updates, new_value_state =
        value_opt.update !value_opt_state value_params value_grads in
      value_opt_state := new_value_state;
      Kaun.Optimizer.apply_updates_inplace value_params value_updates;

      (* Update actor with batched computation *)
      let policy_loss, policy_grads = Kaun.value_and_grad (fun pp ->
        (* Process all states at once *)
        let all_logits = Kaun.apply policy_net pp ~training:true all_states in
        let all_log_probs = Slide2.log_softmax ~axis:(-1) all_logits in

        (* Convert all actions to one-hot *)
        let action_indices = Array.init n_actions (fun i ->
          Rune.astype Rune.int32 episode_data.actions.(i)
        ) in

        let actions_one_hot_list = List.map (fun action_idx ->
          let one_hot = Rune.one_hot ~num_classes:4 action_idx in
          Rune.astype Rune.float32 one_hot
        ) (Array.to_list action_indices) in

        let all_actions_one_hot = Rune.stack ~axis:0 actions_one_hot_list in

        (* Select log probs for taken actions *)
        let selected_log_probs =
          Rune.sum ~axes:[|1|] (Rune.mul all_actions_one_hot all_log_probs) in

        (* Weight by advantages *)
        let advantages_tensor =
          Rune.init Rune.c Rune.float32 [|n_actions|] (fun idxs ->
            advantages.(idxs.(0))
          ) in

        (* Policy gradient loss *)
        let policy_losses = Rune.mul (Rune.neg selected_log_probs) advantages_tensor in
        Rune.mean policy_losses
      ) policy_params in

      let policy_updates, new_policy_state =
        policy_opt.update !policy_opt_state policy_params policy_grads in
      policy_opt_state := new_policy_state;
      Kaun.Optimizer.apply_updates_inplace policy_params policy_updates;

      (* Record metrics *)
      history_returns.(episode - 1) <- Array.fold_left (+.) 0.0 episode_data.rewards;
      history_actor_losses.(episode - 1) <- Rune.item [] policy_loss;
      history_critic_losses.(episode - 1) <- Rune.item [] value_loss;
    end else begin
      (* Empty episode *)
      history_returns.(episode - 1) <- 0.0;
      history_actor_losses.(episode - 1) <- 0.0;
      history_critic_losses.(episode - 1) <- 0.0;
    end
  done;

  (* Return training history with critic losses as main losses *)
  {
    returns = history_returns;
    losses = history_critic_losses;  (* Use critic losses as the main loss metric *)
    collected_episodes = List.rev !collected_episodes;
  }