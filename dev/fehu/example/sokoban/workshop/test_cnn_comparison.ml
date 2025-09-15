(*
Test and compare CNN-based networks vs fully-connected networks
*)

open Workshop
open Exercise4_cnn

(* Test CNN shapes *)
let test_shapes () =
  Printf.printf "Testing CNN network shapes...\n";
  test_cnn_shapes ();
  Printf.printf "✓ All shape tests passed!\n\n"

(* Compare training performance *)
let compare_algorithms () =
  Printf.printf "Comparing CNN vs FC networks on Sokoban...\n\n";

  (* Create environment *)
  let env = Slide1.create_simple_gridworld 5 in

  (* Training parameters *)
  let n_episodes = 100 in
  let learning_rate = 0.001 in
  let gamma = 0.99 in

  Printf.printf "Training REINFORCE with FC network...\n";
  let _policy_net, _params, _episodes, fc_history =
    Slide4_batched.train_reinforce env n_episodes learning_rate gamma () in

  Printf.printf "Training REINFORCE with CNN network...\n";
  let cnn_history = Slide4_cnn.train_reinforce_cnn env n_episodes learning_rate gamma () in

  (* Compare results *)
  Printf.printf "\n=== Performance Comparison ===\n";

  (* Average returns for last 10 episodes *)
  let last_n = min 10 n_episodes in
  let fc_last_returns = Array.sub fc_history.returns (n_episodes - last_n) last_n in
  let cnn_last_returns = Array.sub cnn_history.returns (n_episodes - last_n) last_n in

  let fc_avg = Array.fold_left (+.) 0.0 fc_last_returns /. float_of_int last_n in
  let cnn_avg = Array.fold_left (+.) 0.0 cnn_last_returns /. float_of_int last_n in

  Printf.printf "FC Network - Average return (last %d episodes): %.2f\n" last_n fc_avg;
  Printf.printf "CNN Network - Average return (last %d episodes): %.2f\n" last_n cnn_avg;

  (* Max return achieved *)
  let fc_max = Array.fold_left max neg_infinity fc_history.returns in
  let cnn_max = Array.fold_left max neg_infinity cnn_history.returns in

  Printf.printf "\nFC Network - Max return: %.2f\n" fc_max;
  Printf.printf "CNN Network - Max return: %.2f\n" cnn_max;

  (* First episode to reach positive return *)
  let find_first_positive returns =
    let rec loop i =
      if i >= Array.length returns then None
      else if returns.(i) > 0.0 then Some (i + 1)
      else loop (i + 1)
    in
    loop 0
  in

  let fc_first_pos = find_first_positive fc_history.returns in
  let cnn_first_pos = find_first_positive cnn_history.returns in

  Printf.printf "\nFC Network - First positive return: %s\n"
    (match fc_first_pos with None -> "Never" | Some ep -> Printf.sprintf "Episode %d" ep);
  Printf.printf "CNN Network - First positive return: %s\n"
    (match cnn_first_pos with None -> "Never" | Some ep -> Printf.sprintf "Episode %d" ep);

  (* Print sample trajectory *)
  Printf.printf "\n=== Sample Episodes ===\n";
  if List.length cnn_history.collected_episodes > 0 then begin
    let last_episode = List.hd cnn_history.collected_episodes in
    Printf.printf "CNN final episode: %d steps, total reward: %.1f\n"
      (Array.length last_episode.actions)
      (Array.fold_left (+.) 0.0 last_episode.rewards)
  end

(* Test A2C with CNN *)
let test_a2c_cnn () =
  Printf.printf "\nTesting A2C with CNN networks...\n";

  let env = Slide1.create_simple_gridworld 5 in
  let n_episodes = 50 in
  let lr_actor = 0.001 in
  let lr_critic = 0.003 in
  let gamma = 0.99 in

  let history = Slide6_cnn.train_actor_critic_cnn env n_episodes lr_actor lr_critic gamma () in

  let last_10_avg =
    let last_n = min 10 n_episodes in
    let last_returns = Array.sub history.returns (n_episodes - last_n) last_n in
    Array.fold_left (+.) 0.0 last_returns /. float_of_int last_n
  in

  Printf.printf "A2C-CNN Average return (last 10 episodes): %.2f\n" last_10_avg;
  Printf.printf "A2C-CNN Max return: %.2f\n" (Array.fold_left max neg_infinity history.returns);
  Printf.printf "✓ A2C-CNN training completed successfully!\n"

(* Test PPO with CNN *)
let test_ppo_cnn () =
  Printf.printf "\nTesting PPO with CNN networks...\n";

  let env = Slide1.create_simple_gridworld 5 in
  let n_episodes = 50 in
  let learning_rate = 0.001 in
  let gamma = 0.99 in
  let epsilon = 0.2 in
  let beta = 0.01 in

  let history = Slide9_cnn.train_reinforce_plus_plus_cnn env n_episodes learning_rate gamma epsilon beta () in

  let last_10_avg =
    let last_n = min 10 n_episodes in
    let last_returns = Array.sub history.returns (n_episodes - last_n) last_n in
    Array.fold_left (+.) 0.0 last_returns /. float_of_int last_n
  in

  Printf.printf "PPO-CNN Average return (last 10 episodes): %.2f\n" last_10_avg;
  Printf.printf "PPO-CNN Max return: %.2f\n" (Array.fold_left max neg_infinity history.returns);
  Printf.printf "✓ PPO-CNN training completed successfully!\n"

(* Main test function *)
let () =
  Random.self_init ();
  test_shapes ();
  compare_algorithms ();
  test_a2c_cnn ();
  test_ppo_cnn ();
  Printf.printf "\n✅ All CNN tests completed successfully!\n"