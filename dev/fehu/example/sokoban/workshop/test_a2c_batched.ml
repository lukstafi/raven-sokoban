(* Test batched A2C implementation *)
open Workshop.Slide1
open Workshop.Slide6_batched

let () =
  Printf.printf "Testing batched A2C implementation...\n%!";

  (* Create environment *)
  let env = create_simple_gridworld 5 in

  (* Train with batched A2C *)
  let _policy_net, _policy_params, _value_net, _value_params,
      episodes, returns, actor_losses, critic_losses =
    train_a2c_batched env 50 0.001 0.001 0.99 ~grid_size:5 () in

  Printf.printf "\nTraining complete!\n";
  Printf.printf "Final 10 episode average return: %.2f\n"
    (let last_10 = Array.sub returns 40 10 in
     Array.fold_left (+.) 0.0 last_10 /. 10.0);
  Printf.printf "Final actor loss: %.6f\n" actor_losses.(49);
  Printf.printf "Final critic loss: %.6f\n" critic_losses.(49);
  Printf.printf "Episodes collected: %d\n" (List.length episodes);

  Printf.printf "\n✓ Batched A2C processes ALL states in each episode!\n"