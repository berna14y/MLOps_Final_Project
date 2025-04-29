# main.py
from EDA import perform_eda
from preprocess import prepare_data
from train_model import train_and_evaluate, analyze_predictions, explain_model
from utils import plot_feature_importance

def main():
    # Step 1: Perform EDA
    print("🚀 Performing Exploratory Data Analysis...")
    df = perform_eda()
    
    # Step 2: Preprocess data
    print("\n🔧 Preprocessing data...")
    data_dict = prepare_data()
    
    # Step 3: Train and evaluate model
    print("\n🤖 Training and evaluating model...")
    model = train_and_evaluate(
        data_dict['X_train'], 
        data_dict['y_train'], 
        data_dict['X_test'], 
        data_dict['y_test']
    )
    
    # Step 4: Analyze predictions
    print("\n📊 Analyzing predictions...")
    analyze_predictions(model, data_dict['X_test'], data_dict['y_test'])
    
    # Step 5: Explain model
    print("\n🔍 Explaining model predictions...")
    explain_model(model, data_dict['X_train'], data_dict['X_test'], model_type='xgb')
    
    # Step 6: Plot feature importance
    print("\n🏆 Plotting feature importance...")
    feature_imp = plot_feature_importance(model, data_dict['X_train'].columns)
    print(feature_imp)

if __name__ == "__main__":
    main()