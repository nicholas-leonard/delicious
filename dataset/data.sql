--Choose a dense cluster 
SELECT cluster_key, sum(density)
FROM hyper.cluster2
GROUP BY cluster_key
ORDER BY sum DESC

--What do its page titles look like?
--111056 mac
--111082 wordpress
SELECT a.page_title
FROM web.page_info AS a, hyper.cluster2 AS c2, hyper.cluster3 AS c3, hyper.cluster4 AS c4, hyper.cluster5 AS c5
WHERE c2.cluster_key = 111082 AND c2.item_key = c3.cluster_key AND c3.item_key = c4.cluster_key AND c4.item_key = c5.cluster_key AND c5.item_key = a.page_id

CREATE SCHEMA webcluster;

CREATE TABLE webcluster.corpus (
	doc_id integer NOT NULL,
	doc_url character varying,
	doc_main_type character varying(255),
	doc_sub_type character varying(255),
	doc_protocol character varying(255),
	doc_language character varying(255),
	doc_description character varying(2000),
	doc_title character varying(1000),
	PRIMARY KEY(doc_id)
);

INSERT INTO webcluster.corpus (doc_id, doc_url, doc_main_type, doc_sub_type, doc_protocol, doc_language, doc_description, doc_title)  (
	SELECT page_id, page_url, page_main_type, page_sub_type, page_protocol, page_language, page_description, page_title
	FROM web.page_info AS a, hyper.cluster2 AS c2, hyper.cluster3 AS c3, hyper.cluster4 AS c4, hyper.cluster5 AS c5
	WHERE c2.cluster_key = 111082 AND c2.item_key = c3.cluster_key AND c3.item_key = c4.cluster_key AND c4.item_key = c5.cluster_key AND c5.item_key = a.page_id
);

--How many features? 70k in 3 pages or more. 60k in 4 pages or more. 210k features in all

SELECT feature_key, COUNT(*)
FROM hyper.feature_input AS a, hyper.cluster2 AS c2, hyper.cluster3 AS c3, hyper.cluster4 AS c4, hyper.cluster5 AS c5
WHERE c2.cluster_key = 111082 AND c2.item_key = c3.cluster_key AND c3.item_key = c4.cluster_key AND c4.item_key = c5.cluster_key AND c5.item_key = a.page_id
GROUP BY feature_key
ORDER BY count DESC

CREATE TABLE webcluster.document_term_temp (
	doc_id		INT4,
	term_key	INT4,
	term_frequency	FLOAT8,
	PRIMARY KEY (doc_id, term_key)
);

INSERT INTO webcluster.document_term_temp (doc_id, term_key, term_frequency) (
	SELECT doc_id, feature_key, feature_freq
	FROM webcluster.corpus AS a, hyper.feature_input AS b
	WHERE a.doc_id = b.page_id
	ORDER BY doc_id ASC, feature_freq DESC
); --5,395,612 rows affected

--DROP TABLE webcluster.term;
CREATE TABLE webcluster.term (
	term_key	INT4,
	term_string	VARCHAR(255),
	term_doc_count	INT4,
	PRIMARY KEY (term_key)
);

INSERT INTO webcluster.term (
	SELECT term_key, feature_string, count
	FROM	(
		SELECT term_key, COUNT(*)
		FROM webcluster.document_term_temp
		GROUP BY term_key 
		) AS a, web.feature AS b
	WHERE a.term_key = b.feature_key AND count >= 100
	ORDER BY count DESC
);--6275 rows affected

--DROP TABLE webcluster.document_term;
CREATE TABLE webcluster.document_term (
	doc_id		INT4,
	term_key	INT4,
	term_frequency  FLOAT8,
	PRIMARY KEY (doc_id, term_key)
);

INSERT INTO webcluster.document_term (doc_id, term_key, term_frequency) (
	SELECT doc_id, term_key, term_frequency
	FROM webcluster.document_term_temp AS a
	WHERE (SELECT b.term_key FROM webcluster.term AS b WHERE a.term_key = b.term_key LIMIT 1) IS NOT NULL
	ORDER BY doc_id ASC, term_frequency DESC
);--4,230,065 rows affected rows affected

CREATE TABLE webcluster.document_term_sum_temp (
	doc_id	INT4,
	term_frequency_sum FLOAT8,
	PRIMARY KEY (doc_id)
);

INSERT INTO webcluster.document_term_sum_temp (
	SELECT doc_id, SUM(term_frequency)
	FROM webcluster.document_term
	GROUP BY doc_id 
);

CREATE TABLE webcluster.document_term_input (
	doc_id			INT4,
	term_key_array		INT4[],
	term_frequency_array	FLOAT8[],
	PRIMARY KEY (doc_id)
);

INSERT INTO webcluster.document_term_input (doc_id, term_key_array, term_frequency_array) (
	SELECT doc_id, array_agg(term_key), array_agg(term_freq)
	FROM	(
		SELECT b.doc_id, b.term_key, b.term_frequency/a.term_frequency_sum AS term_freq
		FROM webcluster.document_term_sum_temp AS a, webcluster.document_term AS b
		WHERE a.doc_id = b.doc_id
		) AS a
	GROUP BY doc_id
);

-- All documents still have terms:
SELECT COUNT(*), MIN(array_length(term_key_array, 1)),  AVG(array_length(term_key_array, 1)), MAX(array_length(term_key_array, 1))
FROM webcluster.document_term_input AS a
--10000;13;423.0065000000000000;3100

/* Clusters as one-hot class labels */
CREATE TABLE webcluster.doc_cluster (
	doc_id			INT4,
	cluster10_key		INT4,
	cluster100_key		INT4,
	cluster1000_key		INT4,
	PRIMARY KEY (doc_id)
);

INSERT INTO webcluster.doc_cluster (doc_id, cluster10_key, cluster100_key, cluster1000_key) (
	SELECT a.doc_id, c5.cluster_key, c4.cluster_key, c3.cluster_key
	FROM webcluster.corpus AS a, hyper.cluster2 AS c2, hyper.cluster3 AS c3, hyper.cluster4 AS c4, hyper.cluster5 AS c5
	WHERE c2.cluster_key = 111082 AND c2.item_key = c3.cluster_key AND c3.item_key = c4.cluster_key AND c4.item_key = c5.cluster_key AND c5.item_key = a.doc_id
); 


UPDATE webcluster.doc_cluster AS u
SET cluster10_key = a.rank, cluster100_key = b.rank, cluster1000_key = c.rank
FROM 	(
	SELECT cluster10_key, rank()
		OVER ( ORDER BY cluster10_key)
	FROM	(
		SELECT DISTINCT cluster10_key
		FROM webcluster.doc_cluster
		) AS a
	) AS a,
	(
	SELECT cluster100_key, rank()
		OVER ( ORDER BY cluster100_key)
	FROM	(
		SELECT DISTINCT cluster100_key
		FROM webcluster.doc_cluster
		) AS a
	) AS b,
	(
	SELECT cluster1000_key, rank()
		OVER ( ORDER BY cluster1000_key)
	FROM	(
		SELECT DISTINCT cluster1000_key
		FROM webcluster.doc_cluster
		) AS a
	) AS c
WHERE u.cluster10_key = a.cluster10_key AND u.cluster100_key = b.cluster100_key AND u.cluster1000_key = c.cluster1000_key

--DROP TABLE webcluster.doc_partition;
CREATE TABLE webcluster.doc_partition (
	doc_id		INT4,
	bucket_id	INT4,
	PRIMARY KEY (doc_id)
);

INSERT INTO webcluster.doc_partition (
	SELECT doc_id, rank()
		OVER ( PARTITION BY cluster10_key ORDER BY doc_id*random() DESC )
	FROM webcluster.doc_cluster AS a
	ORDER BY rank DESC
);

--Identify the corpus with tags:

SELECT tag_name, count, sum
FROM 	(
	SELECT tag_id, COUNT(*), SUM(tag_counts)
	FROM hyper.web_pagetag AS a, webcluster.corpus AS b
	WHERE a.page_id = b.doc_id
	GROUP BY tag_id
	) AS a, web.tag AS b
WHERE a.tag_id = b.tag_id
ORDER BY count DESC
