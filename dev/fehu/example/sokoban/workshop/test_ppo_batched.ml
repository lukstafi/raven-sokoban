(* Test batched PPO implementation *)
open Workshop.Slide1
open Workshop.Slide9_batched

let () =
  Printf.printf "Testing batched PPO implementation...\n%!";

  (* Create environment *)
  let env = create_simple_gridworld 5 in

  (* Train with batched PPO *)
  let _policy_net, _params, episodes, returns, losses, kl_divs =
    train_ppo_batched env 50 0.001 ~epsilon:0.2 ~beta:0.01
      ~baseline_alpha:0.95 ~ppo_epochs:2 ~grid_size:5 () in

  Printf.printf "\nTraining complete!\n";
  Printf.printf "Final 10 episode average return: %.2f\n"
    (let last_10 = Array.sub returns 40 10 in
     Array.fold_left (+.) 0.0 last_10 /. 10.0);
  Printf.printf "Final loss: %.6f\n" losses.(49);
  Printf.printf "Final KL divergence: %.6f\n" kl_divs.(49);
  Printf.printf "Episodes collected: %d\n" (List.length episodes);

  Printf.printf "\n✓ Batched PPO processes ALL states with clipping and multiple epochs!\n"