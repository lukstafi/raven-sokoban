(*
Exercise 4: CNN-based Policy and Value Networks for Sokoban
Replaces fully-connected networks with convolutional architectures
to better capture spatial relationships in the grid.
*)

(* Uses Slide2.sokoban_environment and other utilities *)
open Slide3  (* For episode_data type *)

(* CNN Policy Network *)
let create_cnn_policy_network num_actions =
  Kaun.Layer.sequential [
    (* Input shape: [batch, 1, 5, 5] - single channel for grid values *)

    (* First conv layer: detect basic patterns *)
    (* Note: Kaun conv2d appears to maintain spatial dimensions by default *)
    Kaun.Layer.conv2d ~in_channels:1 ~out_channels:16
                      ~kernel_size:(3, 3) ();
    Kaun.Layer.relu ();

    (* Second conv layer: combine patterns *)
    Kaun.Layer.conv2d ~in_channels:16 ~out_channels:32
                      ~kernel_size:(3, 3) ();
    Kaun.Layer.relu ();

    (* Global pooling or flatten for final layers *)
    Kaun.Layer.flatten ();
    (* Output size with maintained dimensions: 32 * 5 * 5 = 800 *)
    Kaun.Layer.linear ~in_features:800 ~out_features:64 ();
    Kaun.Layer.relu ();

    (* Output layer for actions *)
    Kaun.Layer.linear ~in_features:64 ~out_features:num_actions ();
  ]

(* CNN Value Network *)
let create_cnn_value_network () =
  Kaun.Layer.sequential [
    (* Similar architecture but single value output *)
    Kaun.Layer.conv2d ~in_channels:1 ~out_channels:16
                      ~kernel_size:(3, 3) ();
    Kaun.Layer.relu ();

    Kaun.Layer.conv2d ~in_channels:16 ~out_channels:32
                      ~kernel_size:(3, 3) ();
    Kaun.Layer.relu ();

    (* Flatten instead of pooling for small feature maps *)
    Kaun.Layer.flatten ();

    (* Small FC layer for value estimation - 32*5*5 = 800 *)
    Kaun.Layer.linear ~in_features:800 ~out_features:16 ();
    Kaun.Layer.relu ();
    Kaun.Layer.linear ~in_features:16 ~out_features:1 ();
  ]

(* Initialize CNN policy network *)
let initialize_cnn_policy ?(grid_size=5) () =
  let _ = grid_size in  (* Unused but kept for compatibility *)
  let policy_net = create_cnn_policy_network 4 in
  let device = Rune.c in
  let keys = Array.init 1 (fun _ -> Rune.Rng.key (Random.int 1000000)) in
  let policy_params =
    Kaun.init policy_net ~rngs:keys.(0) ~device
      ~dtype:Rune.float32 in
  (policy_net, policy_params)

(* Initialize CNN value network *)
let initialize_cnn_value ?(grid_size=5) () =
  let _ = grid_size in  (* Unused but kept for compatibility *)
  let value_net = create_cnn_value_network () in
  let device = Rune.c in
  let keys = Array.init 1 (fun _ -> Rune.Rng.key (Random.int 1000000)) in
  let value_params =
    Kaun.init value_net ~rngs:keys.(0) ~device
      ~dtype:Rune.float32 in
  (value_net, value_params)

(* Prepare state for CNN input by adding channel dimension *)
let prepare_state_for_cnn state =
  (* state shape: [5, 5] *)
  (* Add channel dimension: [1, 5, 5] *)
  Rune.reshape [|1; 5; 5|] state

(* Prepare batch of states for CNN *)
let prepare_batch_for_cnn states =
  (* states: array of [5, 5] tensors *)
  (* Stack and add channel: [batch_size, 1, 5, 5] *)
  let states_with_channel = Array.map (fun s ->
    Rune.reshape [|1; 1; 5; 5|] s
  ) states in
  Rune.stack ~axis:0 (Array.to_list states_with_channel)

(* Alternative: prepare states directly in batch format *)
let prepare_states_batch_cnn states_array =
  (* Convert array of [5, 5] tensors to [batch_size, 1, 5, 5] *)
  let n_states = Array.length states_array in
  if n_states = 0 then
    Rune.zeros Rune.c Rune.float32 [|0; 1; 5; 5|]
  else
    let states_list = Array.to_list states_array in
    let states_with_channel = List.map (fun s ->
      (* Each state is [5, 5], reshape to [1, 1, 5, 5] *)
      Rune.reshape [|1; 1; 5; 5|] s
    ) states_list in
    (* Concatenate along batch dimension *)
    Rune.stack ~axis:0 states_with_channel

(* Collect episode with CNN-ready states *)
let collect_episode_cnn env policy_net policy_params max_steps =
  let device = Rune.c in
  let state, _ = env.Fehu.Env.reset () in

  let states = Array.make (max_steps + 1) (Rune.zeros device Rune.float32 [|5; 5|]) in
  let actions = Array.make max_steps (Rune.zeros device Rune.float32 [||]) in
  let rewards = Array.make max_steps 0.0 in
  let log_probs = Array.make max_steps (Rune.zeros device Rune.float32 [||]) in

  states.(0) <- state;

  let rec loop t =
    if t >= max_steps then t
    else
      let current_state = states.(t) in
      (* Add channel dimension for CNN *)
      let state_cnn = Rune.reshape [|1; 1; 5; 5|] current_state in

      (* Get action from policy *)
      let logits = Kaun.apply policy_net policy_params ~training:false state_cnn in
      let probs = Rune.softmax ~axes:[|-1|] logits in
      let log_probs_all = Rune.log (Rune.add probs (Rune.scalar device Rune.float32 1e-8)) in

      (* Sample action from probabilities *)
      let probs_flat = Rune.squeeze ~axes:[|0|] probs in
      let probs_array = Rune.to_array probs_flat in
      let cumsum = Array.make 4 0.0 in
      cumsum.(0) <- probs_array.(0);
      for i = 1 to 3 do
        cumsum.(i) <- cumsum.(i-1) +. probs_array.(i)
      done;
      let r = Random.float 1.0 in
      let action_int = ref 0 in
      for i = 0 to 3 do
        if r > cumsum.(i) then action_int := i + 1
      done;
      let action = Rune.scalar Rune.c Rune.float32 (float_of_int !action_int) in

      (* Get log prob of selected action *)
      let log_probs_array = Rune.to_array (Rune.squeeze ~axes:[|0|] log_probs_all) in
      let action_log_prob = Rune.scalar device Rune.float32 log_probs_array.(!action_int) in

      (* Step environment *)
      let next_state, reward, terminated, truncated, _ = env.Fehu.Env.step action in

      actions.(t) <- action;
      rewards.(t) <- reward;
      log_probs.(t) <- action_log_prob;
      states.(t + 1) <- next_state;

      if terminated || truncated then t + 1
      else loop (t + 1)
  in

  let episode_length = loop 0 in

  (* Trim arrays to actual episode length *)
  {
    states = Array.sub states 0 (episode_length + 1);
    actions = Array.sub actions 0 episode_length;
    rewards = Array.sub rewards 0 episode_length;
    log_probs = Array.sub log_probs 0 episode_length;
  }

(* Test function to verify CNN shapes *)
let test_cnn_shapes () =
  let device = Rune.c in
  let net = create_cnn_policy_network 4 in
  let rng = Rune.Rng.key (Random.int 1000000) in
  let params = Kaun.init net ~rngs:rng ~device ~dtype:Rune.float32 in

  (* Test single state *)
  let state = Rune.randn device Rune.float32 [|1; 1; 5; 5|] in
  let output = Kaun.apply net params ~training:false state in
  assert (Rune.shape output = [|1; 4|]);  (* batch=1, actions=4 *)

  (* Test batch *)
  let batch = Rune.randn device Rune.float32 [|32; 1; 5; 5|] in
  let output = Kaun.apply net params ~training:false batch in
  assert (Rune.shape output = [|32; 4|]);  (* batch=32, actions=4 *)

  Printf.printf "CNN shape tests passed!\n"

(* Advanced: Residual block for deeper networks - simplified *)
(* Note: This is conceptual - actual implementation would need proper layer composition *)
let conv_residual_block _in_channels _out_channels =
  (* For now, just return a regular conv block *)
  Kaun.Layer.sequential [
    Kaun.Layer.conv2d ~in_channels:32 ~out_channels:32
                      ~kernel_size:(3, 3) ();
    Kaun.Layer.relu ();
  ]

(* More sophisticated CNN policy with residual connections *)
let create_resnet_policy_network num_actions =
  Kaun.Layer.sequential [
    (* Initial convolution *)
    Kaun.Layer.conv2d ~in_channels:1 ~out_channels:16
                      ~kernel_size:(3, 3) ();
    Kaun.Layer.relu ();

    (* Residual blocks would go here - simplified for now *)
    Kaun.Layer.conv2d ~in_channels:16 ~out_channels:32
                      ~kernel_size:(3, 3) ();
    Kaun.Layer.relu ();

    (* Skip the third conv to avoid making the feature map too small *)
    Kaun.Layer.relu ();

    (* Flatten for final layers *)
    Kaun.Layer.flatten ();

    (* Final layers - 32*5*5 = 800 *)
    Kaun.Layer.linear ~in_features:800 ~out_features:64 ();
    Kaun.Layer.relu ();
    Kaun.Layer.linear ~in_features:64 ~out_features:num_actions ();
  ]