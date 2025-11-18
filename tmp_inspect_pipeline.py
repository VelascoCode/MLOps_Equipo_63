import app
import inspect
from pprint import pprint
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

m = app.load_model('models/final_model.pkl')
print('MODEL_TYPE:', type(m))
print('\nfeature_names_in_:', getattr(m, 'feature_names_in_', None))

if hasattr(m, 'named_steps'):
    print('\nNAMED STEPS:')
    for k, v in m.named_steps.items():
        print(' -', k, type(v))
        # transformers_
        try:
            if hasattr(v, 'transformers_'):
                print('   transformers_:')
                for t in v.transformers_:
                    name = t[0]
                    trans = t[1]
                    cols = t[2] if len(t) > 2 else None
                    print('    -', name, type(trans), 'cols=', cols)
        except Exception as e:
            print('   inspect error:', e)
        try:
            if hasattr(v, 'named_transformers_'):
                print('   named_transformers_:', list(getattr(v, 'named_transformers_').keys()))
        except Exception as e:
            print('   inspect error 2:', e)
        try:
            if hasattr(v, 'get_feature_names_out'):
                print('   has get_feature_names_out; sample (first 20):')
                try:
                    out = v.get_feature_names_out()
                    print('    ', out[:20])
                except Exception as e:
                    print('    get_feature_names_out failed:', e)
        except Exception:
            pass

# Inspect ColumnTransformer inside pipeline steps
if isinstance(m, Pipeline):
    for name, step in m.named_steps.items():
        if isinstance(step, ColumnTransformer):
            print('\nFound ColumnTransformer in step:', name)
            try:
                print(' transformers_ keys:', [t[0] for t in step.transformers_])
            except Exception as e:
                print('  error reading transformers_: ', e)

print('\nINSPECTION DONE')
