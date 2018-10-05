import logging
import sqlite3 as lite
import threading
import numpy as np
import os


class DbLogger:
    logsTable = "logs_table"
    runKvStore = "run_kv_store"
    runMetaData = "run_meta_data"
    leafInfoTable = "leaf_info_table"
    runResultsTable = "run_results"
    confusionTable = "confusion_matrices"
    compressionTestsTable = "compression_tests_table"
    embeddingsTable = "embeddings_table"
    trainingSamplesTable = "training_samples"
    trainingSamplesSampledTable = "training_samples_sampled"
    sentimentSamplesTable = "sentiment_training"
    # Tables
    trainingVocabularyLargeTable = "training_vocabulary_large"
    trainingVocabularySmallTable = "training_vocabulary_small"
    testVocabularyLargeTable = "test_vocabulary_large"
    testVocabularySmallTable = "test_vocabulary_small"
    finalizedVocabularyTable = "finalized_vocabulary"

    # Lab
    # log_db_path = "E://BDA//BDA//bda_db.db"

    # Home
    # log_db_path = "D://seq_classification//seq_db.db"

    # Idea GPU
    # log_db_path = "D://deep//seq_classification//seq_db.db"

    # Idea
    # log_db_path = "C://Users//ufuk.bicici//Desktop//BDA//BDA//bda_db.db"

    # DGX DB
    log_db_path = "/raid/users/ucbicici/Code/seq_classification/seq_db.db"


    logFormatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    loggersDict = {}
    lock = threading.Lock()

    def __init__(self):
        pass

    @staticmethod
    def print_log(log_file_name, log_string):
        print("Enter print_log")
        DbLogger.lock.acquire()
        if not (log_file_name in DbLogger.loggersDict):
            logger_object = logging.getLogger(log_file_name)
            handler = logging.FileHandler(log_file_name, mode="w")
            handler.setFormatter(DbLogger.logFormatter)
            logger_object.setLevel(level=logging.INFO)
            logger_object.addHandler(handler)
            DbLogger.loggersDict[log_file_name] = logger_object
        logger_object = DbLogger.loggersDict[log_file_name]
        logger_object.info(log_string)
        print(log_string)
        DbLogger.lock.release()
        print("Exit print_log")

    @staticmethod
    def get_run_id():
        print("Enter get_run_id")
        DbLogger.lock.acquire()
        con = lite.connect(DbLogger.log_db_path)
        curr_id = None
        with con:
            cur = con.cursor()
            cur.execute("SELECT MAX(RunId) + 1 AS CurrId FROM {0}".format(DbLogger.runMetaData))
            rows = cur.fetchall()
            for row in rows:
                if row[0] is None:
                    curr_id = 0
                else:
                    curr_id = row[0]
        DbLogger.lock.release()
        print("Exit get_run_id")
        return curr_id

    @staticmethod
    def read_confusion_matrix(run_id, dataset, iteration, num_of_labels, leaf_id):
        DbLogger.lock.acquire()
        con = lite.connect(DbLogger.log_db_path)
        cm = np.zeros(shape=(num_of_labels, num_of_labels))
        with con:
            cur = con.cursor()
            sql_command = "SELECT * FROM {0} WHERE RunId={1} AND Iteration={2} AND Dataset={3} AND LeafIndex={4}" \
                .format(DbLogger.confusionTable, run_id, iteration, dataset, leaf_id)
            cur.execute(sql_command)
            rows = cur.fetchall()
            for row in rows:
                true_label = row[4]
                predicted_label = row[5]
                frequency = row[6]
                cm[true_label, predicted_label] = frequency
        DbLogger.lock.release()
        return cm

    @staticmethod
    def read_tuples_from_table(table_name, condition=None):
        DbLogger.lock.acquire()
        con = lite.connect(DbLogger.log_db_path)
        with con:
            cur = con.cursor()
            sql_command = "SELECT * FROM {0}".format(table_name)
            if condition is not None:
                sql_command = "{0} WHERE {1}".format(sql_command, condition)
            cur.execute(sql_command)
            rows = cur.fetchall()
        DbLogger.lock.release()
        return rows

    @staticmethod
    def read_test_accuracy(run_id, type):
        DbLogger.lock.acquire()
        con = lite.connect(DbLogger.log_db_path)
        accuracy = None
        with con:
            cur = con.cursor()
            sql_command = "SELECT TestAccuracy FROM {0} WHERE RunId={1} AND Type={2}" \
                .format(DbLogger.runResultsTable, run_id, type)
            cur.execute(sql_command)
            rows = cur.fetchall()
            for row in rows:
                accuracy = float(row[0])
        DbLogger.lock.release()
        return accuracy

    @staticmethod
    def log_bnn_explanation(runId, explanation_string):
        print("Enter log_bnn_explanation")
        DbLogger.lock.acquire()
        explanation_string = explanation_string.replace("'", "")
        con = lite.connect(DbLogger.log_db_path)
        with con:
            cur = con.cursor()
            cur.execute("INSERT INTO {0} VALUES({1},'{2}')".format(DbLogger.runMetaData, runId, explanation_string))
        DbLogger.lock.release()
        print("Exit log_bnn_explanation")

    @staticmethod
    def log_kv_store(kv_store_rows):
        print("Enter log_kv_store")
        DbLogger.lock.acquire()
        kv_store_rows_as_tuple = tuple(kv_store_rows)
        con = lite.connect(DbLogger.log_db_path)
        with con:
            cur = con.cursor()
            cur.executemany("INSERT INTO {0} VALUES(?, ?, ?, ?, ?)"
                            .format(DbLogger.runKvStore), kv_store_rows_as_tuple)
        DbLogger.lock.release()
        print("Exit log_kv_store")

    @staticmethod
    def log_into_log_table(log_table_rows):
        print("Enter log_into_log_table")
        DbLogger.lock.acquire()
        log_table_rows_as_tuple = tuple(log_table_rows)
        con = lite.connect(DbLogger.log_db_path)
        with con:
            cur = con.cursor()
            cur.executemany("INSERT INTO {0} VALUES(?, ?, ?, ?, ?, ?, ?, ?)"
                            .format(DbLogger.logsTable), log_table_rows_as_tuple)
        DbLogger.lock.release()
        print("Exit log_into_log_table")

    @staticmethod
    def log_into_leaf_table(log_table_rows):
        print("Enter log_into_leaf_table")
        DbLogger.lock.acquire()
        log_table_rows_as_tuple = tuple(log_table_rows)
        con = lite.connect(DbLogger.log_db_path)
        with con:
            cur = con.cursor()
            cur.executemany("INSERT INTO {0} VALUES(?, ?, ?)"
                            .format(DbLogger.leafInfoTable), log_table_rows_as_tuple)
        DbLogger.lock.release()
        print("Exit log_into_leaf_table")

    @staticmethod
    def write_into_table(rows, table, col_count):
        print("Enter write_into_table")
        DbLogger.lock.acquire()
        rows_as_tuple = tuple(rows)
        con = lite.connect(DbLogger.log_db_path)
        with con:
            cur = con.cursor()
            insert_cmd = "INSERT INTO {0} VALUES(".format(table)
            for i in range(0, col_count):
                insert_cmd += "?"
                if i != col_count - 1:
                    insert_cmd += ", "
            insert_cmd += ")"
            cur.executemany(insert_cmd, rows_as_tuple)
        DbLogger.lock.release()
        print("Exit write_into_table")

    @staticmethod
    def create_index(index_name, table, columns):
        DbLogger.lock.acquire()
        con = lite.connect(DbLogger.log_db_path)
        with con:
            cur = con.cursor()
            cmd = "CREATE INDEX {0} ON {1} (".format(index_name, table)
            for index, column in enumerate(columns):
                cmd += column
                if index != len(columns)-1:
                    cmd += ","
            cmd += ")"
            cur.execute(cmd)
        DbLogger.lock.release()

    @staticmethod
    def delete_table(table):
        DbLogger.lock.acquire()
        con = lite.connect(DbLogger.log_db_path)
        with con:
            cur = con.cursor()
            cmd = "DELETE FROM {0}".format(table)
            cur.execute(cmd)
        DbLogger.lock.release()

    @staticmethod
    def does_rows_exist(columns_to_values, table):
        print("Enter does_rows_exist")
        DbLogger.lock.acquire()
        sql_command = "SELECT * FROM {0} WHERE ".format(table)
        for item in enumerate(columns_to_values.items()):
            index = item[0]
            col_name = item[1][0]
            value = item[1][1]
            sql_command += "{0} = {1}".format(col_name, value)
            if index != len(columns_to_values) - 1:
                sql_command += " AND "
        con = lite.connect(DbLogger.log_db_path)
        with con:
            cur = con.cursor()
            cur.execute(sql_command)
            rows = cur.fetchall()
            does_exist = len(rows) > 0
        DbLogger.lock.release()
        print("Exit does_rows_exist")
        return does_exist
