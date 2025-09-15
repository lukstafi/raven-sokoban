(* Run 4 Batched: REINFORCE training with all states processed *)
open Workshop.Slide1
open Workshop.Exercise1_solution

let () =
  print_endline "\n==== Workshop Run 4 Batched: REINFORCE with All States ====\n";

  (* Create environment *)
  let env = create_simple_gridworld 5 in

  (* Train the agent with batched implementation *)
  let history = train_reinforce_batched env 500 0.01 0.99 ~grid_size:5 () in

  (* Show final statistics *)
  Printf.printf "\nTraining complete!\n";
  Printf.printf "Final 10 episode average return: %.2f\n"
    (let last_10 = Array.sub history.returns 490 10 in
     Array.fold_left (+.) 0.0 last_10 /. 10.0);
  Printf.printf "Episodes with goal reached: %d / 500\n"
    (Array.fold_left (fun acc r -> if r > 5.0 then acc + 1 else acc) 0 history.returns);

  (* Visualize first and last collected episodes *)
  (match history.collected_episodes with
  | [] -> print_endline "No episodes collected!"
  | [single] -> Workshop.Helpers.visualize_episode single 1
  | first :: rest ->
      let last = List.hd (List.rev rest) in
      print_endline "\n=== First Episode ===";
      Workshop.Helpers.visualize_episode first 1;
      print_endline "\n=== Last Episode ===";
      Workshop.Helpers.visualize_episode last 500);

  print_endline "\n✓ Batched REINFORCE processes ALL states in each episode!";
  print_endline "  This leads to faster learning and better sample efficiency."