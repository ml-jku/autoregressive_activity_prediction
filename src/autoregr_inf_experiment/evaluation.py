"""
This file evaluates the the autoregressive inference experiment.
"""

#---------------------------------------------------------------------------------------
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------
def evaluate_experiment(experiment_name):
    #-----------------------------------------------------------------------------------
    # Helper functions
    def load_results(experiment_name):
        current_loc = __file__.rsplit("/",3)[0]
        if experiment_name == 'semi_supervised':
            with open(current_loc + '/results/autoreg_inf_fsmol.pkl',
                    'rb') as f:
                aucs, dauc_prs = pickle.load(f)
        elif experiment_name == 'transductive':
            with open(current_loc + '/results/transductive_inf_fsmol.pkl',
                    'rb') as f:
                aucs, dauc_prs = pickle.load(f)
        elif experiment_name == 'semi_supervised_baseline':
            with open(current_loc + '/results/autoreg_inf_baseline_fsmol.pkl',
                    'rb') as f:
                aucs, dauc_prs = pickle.load(f)
        elif experiment_name == 'semi_supervised_protonet':
            with open(current_loc + '/results/protonet_autoreg_inf_fsmol.pkl',
                    'rb') as f:
                aucs, dauc_prs = pickle.load(f)
        else:
            raise ValueError('Invalid experiment name')
        
        return aucs, dauc_prs
    
    def load_task_categories():
        """
        This function loads the task  category information provided in the FS-Mol repo
        and links it to the task ids.
        """
        
        # Load dictionary which links task names to task ids
        current_loc = __file__.rsplit("/",3)[0]
        with open(current_loc + '/assets/fs_mol_data/dict_task_names_id.pkl',
                  'rb') as f:
            task_dict = pickle.load(f)
        
        # Load task information
        task_info = pd.read_csv(current_loc + '/assets/fs_mol_data/test_proteins.csv')
        task_info['category'] = task_info['EC_super_class_name'].str.replace(
            'transferase', 'kinase')
        task_info = task_info[['chembl_id', 'category']]
        
        # Add task ids to dataframe
        task_info['task_id'] = [task_dict[chembl_id] for chembl_id 
                                in task_info.chembl_id]
        
        # create task categry dictionary
        task_categories = {task_id: category for task_id, category in
                           zip(task_info.task_id, task_info.category)}
        return task_categories
        
    def load_fsmol_main_results():
        current_loc = __file__.rsplit("/",3)[0]
        df = pd.read_csv(current_loc +
                         '/assets/performance_on_fsmol/performance_test_tasks.csv')
        return df
    
    def create_performance_array(performance_dict:dict) -> np.array:
        """
        The raw results are stored in a nested dictionary where the first level of keys
        corresponds to the task id and the second level to the reruns with different
        seeds. For each (task-id, rerun) pair, the performance values are stored in a
        list.
        This function creates an array with the performance values:
        - The first dimensions corresponds to the task id
        - The second dimension corresponds to the reruns
        - The third dimension corresponds to the different support set sizes
        """
        
        performance_values = np.zeros((157,10,32))
        # Compute mean performaces across reruns
        for task_nbr, task_id in enumerate(performance_dict.keys()):
            # Filter relevant values
            rerun_dict = performance_dict[task_id]
            
            for rerun_nbr,rerun_key in enumerate(rerun_dict.keys()):
                values = rerun_dict[rerun_key]
                
                performance_values[task_nbr,rerun_nbr,:] = values
        
        return performance_values
                                  
    def plot_scatterplot_performance_autoreggain(performance_values:np.array,
                                                 metric:['AUC', 'ΔAUC-PR'],
                                                 general_performance:pd.DataFrame,
                                                 task_categories_dict:dict,
                                                 experiment_name=str,
                                                 iteration_round=8):
        
        # performance values dims [157, 10, 32]
        reference_values = performance_values[:,:,[0]]
        assert reference_values.shape == (157,10,1)
        reference_values = np.broadcast_to(reference_values, (157,10,32))
        
        relative_gains = performance_values - reference_values
        
        # compute means and stds across reruns
        mean_gains = np.mean(relative_gains, axis=1)[:, iteration_round]
        std_gains = np.std(relative_gains, axis=1)[:, iteration_round]
        assert mean_gains.shape == (157,)
        
        
        # task groups
        task_categories = np.array(
            [task_categories_dict[task_id] for task_id in range(157)]
            )
        

        
        # prepare plot
        if metric == 'AUC':
            x_values = general_performance.auc.values.flatten()
        elif metric == 'ΔAUC-PR':
            x_values = general_performance.daucprs.values.flatten()
            
        y_values = mean_gains
        
        plt.figure(figsize=(4, 4))
        for task_category in np.unique(task_categories):
            plt.scatter(x_values[task_categories == task_category],
                        y_values[task_categories == task_category],
                        label=task_category,
                        marker='.')
        plt.grid()
        plt.legend(fontsize=8)
        plt.xlabel(f'Performance [{metric}] on FS-Mol main exp.')
        plt.ylabel(f'Gain [{metric}] ({iteration_round} autoregr. inf. iterations)')
    
        current_loc = __file__.rsplit("/",3)[0]
        plt.savefig(current_loc +
                    (f'/results/plots/{experiment_name}/scatterplot_performance_gain_'
                     f'{metric}.pdf'),
                        bbox_inches='tight', pad_inches=0.1)
        plt.close()
            
    def plot_mean_performance_across_tasks(performance_values:np.array,
                                                metric:['AUC', 'ΔAUC-PR'],
                                                experiment_name):
        """
        This functions evalueates the performance of the model across all test tasks
        (analogous to the FS-Mol main benchmark experiment).
        mean performance:
        * Firstly the mean for a task is computed across the different reruns
        * Eventually the mean performance across tasks is computed
        """
        
        # Compute mean performance across reruns
        mean_rerun_performance = np.mean(performance_values, axis=1)
        
        # Compute mean and std across reruns
        mean_performance = np.mean(mean_rerun_performance, axis=0)
        std_performance = np.std(mean_rerun_performance, axis=0)
        
        # Create plot
        title = f"MHNfs performance [{metric}] with increasing support set"
        x_label = 'Nbr of both actives and inactives in the support set'
        y_label = f'{metric} (mean across test tasks)'
        
        x_values = np.arange(1, len(mean_performance)+1, 1)
        y_values = mean_performance
        y_upper_bound = mean_performance + std_performance
        y_lower_bound = mean_performance - std_performance
        
        plt.figure(figsize=(15,7))
        plt.plot(x_values, y_values, color='tab:blue')
        plt.fill_between(x_values,
                         y_lower_bound,
                         y_upper_bound,
                         color='tab:blue',
                         alpha=0.3)
        plt.grid()
        plt.xticks(np.arange(1, len(mean_performance)+1, 1))
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        current_loc = __file__.rsplit("/",3)[0]
        plt.savefig(current_loc +
                    f'/results/plots/{experiment_name}/'
                    f'{metric}_mean_performance_across_tasks.pdf',
                    bbox_inches='tight',pad_inches=0.1)
        
    def plot_mean_performance_across__reruns(
            performance_values:np.array,                                   
            metric:['AUC', 'ΔAUC-PR'],
            experiment_name='str'
        ):
        """
        This functions evalueates the performance of the model across all test tasks
        (analogous to the FS-Mol main benchmark experiment).
        mean performance:
        * Firstly the mean across tasks, then the mean across reruns is computed
        """     
        
        # Compute mean performance across tasks
        mean_task_performance = np.mean(performance_values, axis=0)
        
        # Compute mean and std performance across reruns
        mean_performance = np.mean(mean_task_performance, axis=0)
        std_performance = np.std(mean_task_performance, axis=0)
        
        # Create plot
        x_label = 'Nbr of autoregressive inference iterations (1+/1- added in each round)'
        y_label = f'{metric}'
        
        x_values = np.arange(0, len(mean_performance), 1)
        y_values = mean_performance
        y_upper_bound = mean_performance + std_performance
        y_lower_bound = mean_performance - std_performance
        
        plt.figure(figsize=(10,2))
        
        plt.plot(x_values, y_values, color='#f1bc31')
        plt.fill_between(x_values,
                         y_lower_bound,
                         y_upper_bound,
                         color='#f1bc31',
                         alpha=0.3)
        
        x_values = np.arange(0, len(mean_performance), 1)
        y_values = list([mean_performance.flatten()[0]]) * x_values.shape[0] 
        y_upper_bound = list([mean_performance.flatten()[0] + 
                                std_performance.flatten()[0]]) * x_values.shape[0]
        y_lower_bound = list([mean_performance.flatten()[0] -
                             std_performance.flatten()[0]]) * x_values.shape[0]
        
        plt.plot(x_values, y_values, color='#4fb0bf')
        plt.fill_between(x_values,
                         y_lower_bound,
                         y_upper_bound,
                         color='#4fb0bf',
                         alpha=0.3)
        
        plt.grid()
        plt.xticks(np.arange(0, len(mean_performance), 1))
        plt.xlim(0,31)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        current_loc = __file__.rsplit("/",3)[0]
        plt.savefig(current_loc +
                    f'/results/plots/{experiment_name}/'
                    f'{metric}_mean_performance_across_reruns.pdf',
                    bbox_inches='tight',pad_inches=0.1)
         
    def create__performance_table(performace_values:np.array,
                                    metric:['AUC', 'ΔAUC-PR'],
                                    experiment_name=str):
        """
        This function creates a table with the relative gain in performance for each
        support set size.
        """
        # Compute mean performance across tasks
        mean_task_performance = np.mean(performace_values, axis=0)
        
        # Compute mean and std across reruns
        mean_performance = np.mean(mean_task_performance, axis=0)
        std_performance = np.std(mean_task_performance, axis=0)
        
        # Compute relative gain
        reference = np.broadcast_to(
            mean_task_performance[:,0].reshape(-1,1),
            (mean_task_performance.shape[0],32))
        relative_gain = mean_task_performance - reference
        mean_relative_gain = np.mean(relative_gain, axis=0)
        std_relative_gain = np.std(relative_gain, axis=0)
        
        # Create table
        df = pd.DataFrame({'autoreg-iteration':list(range(32)),
                           'autoreg-inf-mean': mean_performance,
                           'autoreg-inf-std': std_performance,
                           'mean-gain': mean_relative_gain,
                           'std-gain': std_relative_gain})
        
        current_loc = __file__.rsplit("/",3)[0]
        df.to_csv(
            current_loc +
            f'/results/performance_tables/{experiment_name}_{metric}.csv')
            
    #-----------------------------------------------------------------------------------
    # Main function
    
    # Load results
    aucs, dauc_prs = load_results(experiment_name=experiment_name)
    
    # Load task categories
    if experiment_name == 'semi_supervised':
        task_categories = load_task_categories()
    
    # create performance arrays
    aucs = create_performance_array(aucs)
    dauc_prs = create_performance_array(dauc_prs)
        
    # Compare autoregressive inf. gain with performance on FS-Mol main experiment
    if experiment_name == 'semi_supervised':
        general_performance = load_fsmol_main_results()
        plot_scatterplot_performance_autoreggain(aucs, 'AUC',
                                                 general_performance,
                                                 task_categories,
                                                 experiment_name)
        plot_scatterplot_performance_autoreggain(dauc_prs, 'ΔAUC-PR',
                                                 general_performance,
                                                 task_categories,
                                                 experiment_name)
        
    # Evaluate how the mean performance of MHNfs changes with increasing support set
    plot_mean_performance_across_tasks(aucs, 'AUC', experiment_name)
    plot_mean_performance_across_tasks(dauc_prs, 'ΔAUC-PR', experiment_name)
    
    plot_mean_performance_across__reruns(aucs, 'AUC', experiment_name)
    plot_mean_performance_across__reruns(dauc_prs, 'ΔAUC-PR', experiment_name)

    # Create performance tables    
    create__performance_table(aucs, 'AUC', experiment_name)
    create__performance_table(dauc_prs, 'ΔAUC-PR', experiment_name)
    
#---------------------------------------------------------------------------------------
if __name__ == '__main__':
    evaluate_experiment(experiment_name='semi_supervised')
    #evaluate_experiment(experiment_name='semi_supervised_baseline')
    #evaluate_experiment(experiment_name='semi_supervised_protonet')
    #evaluate_experiment(experiment_name='transductive')