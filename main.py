import os, requests, random
import sys
import csv
import glob
import pathlib
from custom_gui import visualization_server
from worlds1.world_builder import create_builder
from utils1.util_functions import load_R_to_Py
from pathlib import Path

if __name__ == "__main__":
    print("\nEnter the participant ID:")
    choice1=input()
    print("\nEnter one of the environments 'trial' or 'experiment':")
    choice2=input()
    if choice2=='trial':
        builder = create_builder(exp_version='trial',condition='tutorial')
    else:
        print("\nEnter one of the conditions 'baseline', 'shap', or 'util':")
        choice3=input()
        if choice3=='shap' or choice3=='util' or choice3=='baseline':
            start_scenario = None
            media_folder = pathlib.Path().resolve()
            print("Starting custom visualizer")
            vis_thread = visualization_server.run_matrx_visualizer(verbose=False, media_folder=media_folder)

            robots = ['Brutus'] * 2 + ['Brutus'] * 2
            tasks = [2, 2, 2, 2]
            random.shuffle(robots)
            random.shuffle(tasks)
            print(robots, tasks)

            for i, robot in enumerate(robots, start=0):
                print(f"\nTask {i+1}: The robot is {robot} and task version {tasks[i]}.\n")
                builder = create_builder(id=choice1, exp_version='experiment', name=robot, condition=choice3, task=tasks[i])
                builder.startup(media_folder=media_folder)
                print("Started world...")
                world = builder.get_world()
                builder.api_info['matrx_paused'] = True
                world.run(builder.api_info)

                if choice2=="experiment":
                    fld = os.getcwd()
                    recent_dir = max(glob.glob(os.path.join(fld, '*/'+choice1+'/')), key=os.path.getmtime)
                    recent_dir = max(glob.glob(os.path.join(recent_dir, '*/')), key=os.path.getmtime)
                    action_file = glob.glob(os.path.join(recent_dir,'world_1/action*'))[0]
                    message_file = glob.glob(os.path.join(recent_dir,'world_1/message*'))[0]
                    action_header = []
                    action_contents=[]
                    message_header = []
                    message_contents=[]
                    unique_agent_moves = []
                    with open(action_file) as csvfile:
                        reader = csv.reader(csvfile, delimiter=';', quotechar="'")
                        for row in reader:
                            if action_header==[]:
                                action_header=row
                                continue
                            if row[1:3] not in unique_agent_moves:
                                unique_agent_moves.append(row[1:3])
                            res = {action_header[i]: row[i] for i in range(len(action_header))}
                            action_contents.append(res)
                    
                    with open(message_file) as csvfile:
                        reader = csv.reader(csvfile, delimiter=';', quotechar="'")
                        for row in reader:
                            if message_header==[]:
                                message_header=row
                                continue
                            res = {message_header[i]: row[i] for i in range(len(message_header))}
                            message_contents.append(res)

                    no_messages_human = message_contents[-1]['total_number_messages_human']
                    no_messages_agent = message_contents[-1]['total_number_messages_agent']
                    total_allocations = message_contents[-1]['total_allocations']
                    human_allocations = message_contents[-1]['total_allocations_human']
                    human_interventions = message_contents[-1]['total_interventions']
                    mean_intervention_sensitivity = message_contents[-1]['mean_intervention_sensitivity']
                    disagreement_ratio = int(human_interventions) / (int(total_allocations) - int(human_allocations))
                    no_ticks = action_contents[-1]['tick_nr']
                    completeness = action_contents[-1]['completeness']

                    print("Saving output...")
                    with open(os.path.join(recent_dir,'world_1/output.csv'),mode='w') as csv_file:
                        csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(['completeness','no_ticks','agent_moves','agent_messages','human_messages','allocations','human_allocations','human_interventions', 'disagreement','intervention_sensitivity'])
                        csv_writer.writerow([completeness,no_ticks,len(unique_agent_moves),no_messages_agent,no_messages_human,total_allocations,human_allocations,human_interventions,disagreement_ratio,mean_intervention_sensitivity])
            print("DONE!")
            print("Shutting down custom visualizer")
            r = requests.get("http://localhost:" + str(visualization_server.port) + "/shutdown_visualizer")
            vis_thread.join()
        else:
            print("\nWrong condition name entered")

    builder.stop()