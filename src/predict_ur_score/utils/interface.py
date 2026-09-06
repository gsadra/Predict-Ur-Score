from tabulate import tabulate
from predict_ur_score.ml.predict import Predictor
from predict_ur_score.stats import Stats
from predict_ur_score.ml.eda import EDA
from predict_ur_score.exceptions import PredictorInvalidInput
from typing import Any


class Interface:

    HOME_SELECTION = ['1', '2', '3']
    GENDERS = ['male', 'female']
    ANS = {'yes': True, 'no': False}
    student_inps = {}

    def run(self):
        self.header()

        while True:
            try:
                home_selection = self.home()

                if home_selection == 1:
                    self.predict_sec()
                    self.show_predict()
                elif home_selection == 2:
                    self.stat_sec()
                elif home_selection == 3:
                    break

                input('\nPress Enter to countinue.\n')

            except PredictorInvalidInput as e:
                print(f'Error {e}')
                input('\nPress Enter to try again...\n')
                

    def header(self) -> None:
        print(f'\n *** Predict Ur Score CLI App: v0.1.0 *** \n\n\n')

    def home(self) -> int:
        print('Choose each number that you want to continue: (ex: 1)')
        print(
            '''
            1. Predict Scores.
            2. Check Scores Stats.
            3. exit
            ''')

        return self._check_home_input()

    def predict_sec(self) -> None:
        self.student_inps['gender'] = self._check_gender()
        self.student_inps['part_time_job'] = self._check_part_time_job()
        self.student_inps['absence_days'] = self._check_absence_days()
        self.student_inps['extracurricular_activities'] = self._check_extracurricular_activities()
        self.student_inps['weekly_self_study_hours'] = self._check_weekly_self_study_hours()
        self.student_inps['career_aspiration'] = self._check_career_aspiration()

    def show_predict(self) -> None:
        predictor = Predictor()
        predictions = predictor.predict(student_data=self.student_inps)

        rows = [
            [subject.replace('_', ' ').title(), f'{score:.2f}']
            for subject, score in predictions.items()
        ]

        print('\n\n *** Here are your predicted scores based on the dataset!*** ')
        print(tabulate(rows, headers=["Subject", "Predicted Score"], tablefmt='psql', numalign='center', stralign='center'))
        print('\nBack to home: \n')

    def stat_sec(self) -> None:
        stat = Stats()

        courses = {number + 2: score for number, score in enumerate(stat.TOPICS)}
        courses[1] = 'all_stat'

        print('\nYou can see stats of all scores or each of them.  ')
        for number, score in sorted(courses.items()):
            print(f'  {number}. {score.replace("_", " ").title()}')
        print()

        inp = self._check_stat_input()
        if inp == 1:
            print()
            print(stat.table_format())
            print('\nBack to home:')
        else:
            print()
            print(stat.table_format(courses[inp], all_stats=False))
            print('\nBack to home:')

    def _check_home_input(self) -> int:
        inp = input('Select: ').strip()
        if not inp in self.HOME_SELECTION:
            raise PredictorInvalidInput(f'Input {inp} is not available.')

        return int(inp)
            
    def _check_gender(self) -> str:
        inp = input('\nWhat is your gender? (male, female): ').strip().lower()
        if not inp in self.GENDERS:
            raise PredictorInvalidInput('Type between male or female.')

        return inp

    def _check_part_time_job(self) -> bool:

        inp = input('\nAre you currently on a part-time job? (yes, no): ').strip().lower()
        if not inp in self.ANS:
            raise PredictorInvalidInput('Type between yes or no.')

        return self.ANS[inp]

    def _check_absence_days(self) -> int:
        try:
            inp = int(input('\nHow many days you may be absent? (ex: 4): ').strip())
        except ValueError:
            raise PredictorInvalidInput('Input is not a number!')

        if inp < 0:
            raise PredictorInvalidInput("Absence days can't be negetive!")

        return inp

    def _check_extracurricular_activities(self) -> bool:
        inp = input('\nDo you have extracurricular activities? (yes, no): ').strip().lower()
        if not inp in self.ANS:
            raise PredictorInvalidInput('Type between yes or no.')

        return self.ANS[inp]

    def _check_weekly_self_study_hours(self) -> int:
        try:
            inp = int(input('\nHow many hours your going to study by yourself in a week? (ex: 25): ').strip())
        except ValueError:
            raise PredictorInvalidInput('Input is not a number!')

        if inp < 0:
            raise PredictorInvalidInput("Study hours can't be negetive!")

        return inp

    def _check_career_aspiration(self) -> str:
        eda = EDA()
        career_aspirations = {number + 1: career for number, career in enumerate(eda.career_aspirations())}

        print('\nWhat is your career aspiration based on this list?')
        for number, career in career_aspirations.items():
            print(f'  {number}. {career}')
        print()

        try:
            inp = int(input('Select a number (ex: 3): ').strip())
        except ValueError:
            raise PredictorInvalidInput('Input is not a number!')

        if not inp in career_aspirations.keys():
            raise PredictorInvalidInput('Type between valid numbers.')

        return career_aspirations[inp]

    def _check_stat_input(self) -> int:
        try:
            inp = int(input('Select (ex: 1): '))
        except ValueError:
            raise PredictorInvalidInput('Input is not a number!')
        if inp > len(Stats.TOPICS) + 1:
            raise PredictorInvalidInput('Type between valid numbers.')

        return inp